"""Paired extension of the archived LA-MDSI simulation; writes only in this folder."""
from pathlib import Path
import sys, os, json, hashlib, shutil, ast, types, copy, time, argparse, logging
sys.dont_write_bytecode=True
from project_config import *
BASE=ensure_output(DEPENDENCE)
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr,kendalltau
from threadpoolctl import threadpool_limits
import importlib.util

OLD=LEGACY
SCENARIOS={'null':(0.,0.,.20),'weak_mixed':(.3,.3,.20),
           'autocorrelation_only':(1.2,0.,.20),'heterogeneity_only':(0.,1.2,.20),
           'mixed':(1.2,1.2,.20),'mixed_rare':(1.2,1.2,.05)}
ACTIVE=['weak_mixed','mixed','mixed_rare']
LOADINGS=[0.,.25,.5,.85]
CAPTURES=[0.,.25,.5,.75,1.]
NREPS=100
BOOTSTRAPS=10000
BOOT_SEED=20261010

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,obj):
    p=Path(p); assert p.resolve().is_relative_to(BASE.resolve())
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(df,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(p,index=False,encoding='utf-8-sig')
def std(x): return (x-x.mean())/x.std()
def finite_mean(x):
    v=np.asarray(x,float);return float(v.mean()) if np.isfinite(v).all() else np.nan

def prepare():
    for name in ['source_snapshot','datasets','checkpoints','fold_components','tables','figures','qa']:
        (BASE/name).mkdir(exist_ok=True)
    originals=[CODE/'scientific_core.py',CODE/'legacy_simulation.py',CODE/'historical_core.py',CONFIG/'02_Legacy_simulation.json']
    manifest=BASE/'qa/source_hashes.json'
    if not manifest.exists():
        write_json(manifest,{str(p):sha(p) for p in originals})
        for p in originals:shutil.copy2(p,BASE/'source_snapshot'/p.name)
    cfg=dict(repetitions=NREPS,grid_side=30,sample_size=900,loadings=LOADINGS,captures=CAPTURES,
      scenarios=SCENARIOS,generator_seed_rule='20260824 + original_scenario_index*10000 + repetition',
      driver='S_lambda = standardize(sqrt(1-lambda^2)*S_0 + lambda*H); H unchanged',
      independence='lambda=0 means no template loading in the Gaussian-field generator; realized sitewise correlation need not be zero',
      primary_arm='fixed_amplitude: original a,h coefficients and standardized drivers',
      secondary_arm='variance_matched: common multiplier sqrt(var(a*S0+h*H)/var(a*Slambda+h*H)); only mixed scenarios',
      response='same intercept calibration and Bernoulli law; shared uniforms reproduce original lambda=0 labels exactly',
      folds='archived lambda=0 StratifiedGroupKFold assignment held fixed across dependence and variance arms',
      strata_policies=['frozen: baseline train-only CART/K-means validation labels reused',
                       'adaptive: same train-only algorithms refitted for each generated condition'],
      primary_theta='equal-fold mean of raw/CART theta under frozen strata; historical full-grid fixed-band theta also retained',
      capture='original probability shrinkage p_c=prevalence+c*(true_p-prevalence), with original probability bounds',
      theta='original untruncated centered magnitude ratios, product-complement; no component or score clipping',
      bootstrap=dict(repetitions=BOOTSTRAPS,seed=BOOT_SEED,unit='paired DGP repetition within original scenario',
                     intervals='percentile Monte Carlo confidence intervals; shared resampling across metrics and loading levels'),
      ranking='five capture configurations within scenario; common complete repetition set across all loading levels; median score; numerical tie tolerance 1e-10',
      estimand='response of theta to this generator extension, not bias against a true joint-explanation target',
      scope_limit='driver SD and coefficients are fixed in primary arm; Gaussian marginal shape/covariance kernel and response I/q cannot all be fixed when template loading changes')
    config=BASE/'protocol.json'
    cfg=json.loads(json.dumps(cfg))
    if config.exists(): assert json.loads(config.read_text(encoding='utf-8'))==cfg
    else:write_json(config,cfg)
    import scientific_core as core
    import historical_core as hist
    return core,hist

def check_sources():
    hashes=json.loads((BASE/'qa/source_hashes.json').read_text(encoding='utf-8'))
    changed=[p for p,h in hashes.items() if not Path(p).exists() or sha(p)!=h]
    if changed:raise RuntimeError('Audited original changed: '+str(changed[:3]))
    return len(hashes)

def generated_baseline(hist,index,rep,h,xy):
    seed=20260824+index*10000+rep;rng=np.random.default_rng(seed)
    noise=rng.normal(size=(30,30));s=std(gaussian_filter(noise,sigma=3.,mode='reflect').ravel())
    _,(a,b,prev)=list(SCENARIOS.items())[index]
    signal=a*s+b*h;true=hist.logistic(hist.calibrate_intercept(signal,prev)+signal)
    clone=np.random.default_rng();clone.bit_generator.state=copy.deepcopy(rng.bit_generator.state)
    original_y=clone.binomial(1,true)
    raw=rng.random(len(s));u=np.where(true<=.5,1-raw,raw)
    y=(u<true).astype(int)
    if not np.array_equal(y,original_y):raise RuntimeError('Common-uniform baseline does not reproduce original binomial labels')
    scene=list(SCENARIOS)[index]
    stored=np.load(OLD/'simulation/datasets'/f'{scene}_r{rep:03d}.npz')
    for value,key in [(s,'spatial'),(true,'true_probability'),(y,'response'),(noise,'gaussian_noise')]:
        if not np.array_equal(value,stored[key]):raise RuntimeError(f'Baseline replay differs: {scene} {rep} {key}')
    return seed,s,true,y,u,noise

def diagnostics(hist,core,s0,s,h,a,b,prev,lam,arm,u,w,zone,seed):
    unscaled=a*s+b*h
    scale=1.
    if arm=='variance_matched' and unscaled.var()>0:
        scale=np.sqrt((a*s0+b*h).var()/unscaled.var())
    signal=scale*unscaled;intercept=hist.calibrate_intercept(signal,prev)
    truth=hist.logistic(intercept+signal);y=(u<truth).astype(int)
    # Shapley contrasts of the known additive-logit channels, used only to audit oracle shrinkage.
    p0=hist.logistic(np.full(len(s),intercept))
    pa=hist.logistic(intercept+scale*a*s);ph=hist.logistic(intercept+scale*b*h)
    phi_a=.5*((pa-p0)+(truth-ph));phi_h=.5*((ph-p0)+(truth-pa))
    d=dict(generator_seed=seed,loading=lam,arm=arm,driver_correlation=float(np.corrcoef(s,h)[0,1]),
      baseline_driver_correlation=float(np.corrcoef(s0,h)[0,1]),spatial_mean=float(s.mean()),spatial_sd=float(s.std()),
      heterogeneity_mean=float(h.mean()),heterogeneity_sd=float(h.std()),
      spatial_driver_moran=core.moran(s,w),spatial_driver_q=core.q_stat(s,zone),
      heterogeneity_driver_moran=core.moran(h,w),heterogeneity_driver_q=core.q_stat(h,zone),
      nominal_a=a,nominal_h=b,effective_a=scale*a,effective_h=scale*b,common_amplitude_multiplier=float(scale),
      baseline_signal_variance=float((a*s0+b*h).var()),signal_variance=float(signal.var()),
      target_prevalence=prev,expected_prevalence=float(truth.mean()),observed_prevalence=float(y.mean()),positives=int(y.sum()),
      response_moran=core.moran(y,w),response_q_fixed_bands=core.q_stat(y,zone),
      probability_moran=core.moran(truth,w),probability_q_fixed_bands=core.q_stat(truth,zone),
      shapley_a_sd=float(phi_a.std()),shapley_h_sd=float(phi_h.std()),
      baseline_label_changes=int(np.sum(y!=(u<hist.logistic(hist.calibrate_intercept(a*s0+b*h,prev)+a*s0+b*h)))))
    return d,truth,y,phi_a,phi_h

def pilot(core,hist,xy,h,zone,w):
    rows=[]
    for scene in ACTIVE:
        index=list(SCENARIOS).index(scene);a,b,prev=SCENARIOS[scene]
        for rep in range(1,11):
            seed,s0,p0,y0,u,noise=generated_baseline(hist,index,rep,h,xy)
            for arm in ['fixed_amplitude','variance_matched']:
                for lam in LOADINGS:
                    s=s0.copy() if lam==0 else std(np.sqrt(1-lam**2)*s0+lam*h)
                    d,*_=diagnostics(hist,core,s0,s,h,a,b,prev,lam,arm,u,w,zone,seed)
                    rows.append(dict(scenario=scene,repetition=rep,**d))
    df=pd.DataFrame(rows);csv(df,BASE/'qa/pilot_generator_diagnostics.csv')
    for _,f in df.groupby(['scenario','repetition','arm']):
        assert np.all(np.diff(f.sort_values('loading').driver_correlation)>0)
    assert np.max(abs(df.spatial_sd-1))<1e-12
    assert np.max(abs(df.expected_prevalence-df.target_prevalence))<1e-12
    matched=df.query("arm == 'variance_matched'")
    assert np.max(abs(matched.signal_variance-matched.baseline_signal_variance))<1e-10
    med=df.query("arm == 'fixed_amplitude'").groupby('loading').driver_correlation.median()
    if med.loc[.85]<.75 or med.loc[.85]-med.loc[0]<.35:
        raise RuntimeError('Pilot failed to produce a reliable dependence gradient; no production run')
    write_json(BASE/'qa/pilot_acceptance.json',dict(success=True,rows=len(df),realized_median_correlation=med.to_dict(),
        baseline_replayed_bitwise=True,gradient_monotone_every_repetition=True,expected_prevalence_calibrated=True))

def with_levels(core,initial,y,p,levels):
    out=initial.copy();n=len(y)
    for residual in core.RESIDUALS:
        v=core.residual(y,p,residual);a=out[f'eta_a_{residual}']
        for strata,z in levels.items():
            qy=core.q_stat(y,z);qr=core.q_stat(v,z);eq=(len(np.unique(z))-1)/(n-1)
            hh=core.eta(qy,qr,eq,1-eq)
            out.update({f'q_y_{strata}':qy,f'q_residual_{strata}_{residual}':qr,
                        f'eta_h_{strata}_{residual}':hh,f'theta_{strata}_{residual}':core.joint(a,hh)})
    six=np.array([out[c] for c in core.THETA]);out['n_finite_theta']=int(np.isfinite(six).sum())
    out['r_dsi']=float(np.quantile(six,.25,method='linear')) if np.isfinite(six).all() else np.nan
    out['s_theta']=float(1-(np.quantile(six,.75,method='linear')-np.quantile(six,.25,method='linear'))) if np.isfinite(six).all() else np.nan
    return out

def run(core,hist,pilot_only=False):
    start=time.perf_counter();gy,gx=np.mgrid[:30,:30];xy=np.c_[gx.ravel(),gy.ravel()]
    zone=np.digitize(gx.ravel(),[6,12,18,24]);h=std(np.array([-1.3,.7,-.5,1.4,-.3])[zone])
    w=core.graph(xy,8,'legacy')
    pilot(core,hist,xy,h,zone,w)
    logging.info('Pilot passed: original baseline reproduced; dependency loading gradient reliable')
    if pilot_only:return
    frozen=BASE/'qa/frozen_run_protocol.json'
    if not frozen.exists():
        write_json(frozen,dict(protocol_sha256=sha(BASE/'protocol.json'),runner_sha256=sha(Path(__file__)),
            started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            pilot_passed=True,source_snapshot_sha256={p.name:sha(p) for p in (BASE/'source_snapshot').iterdir() if p.is_file()}))
    else:
        assert json.loads(frozen.read_text(encoding='utf-8'))['protocol_sha256']==sha(BASE/'protocol.json')
    with threadpool_limits(limits=1):
        for index,(scene,(a,b,prev)) in enumerate(SCENARIOS.items()):
            for rep in range(1,NREPS+1):
                key=f'{scene}_r{rep:03d}';checkpoint=BASE/'checkpoints'/f'{key}.csv'
                if checkpoint.exists():continue
                seed,s0,p0,y0,u,noise=generated_baseline(hist,index,rep,h,xy)
                fold=np.load(OLD/'simulation'/f'{key}_fold_assignment.npz')['fold']
                prepared=[]
                x0=np.c_[xy,s0,h]
                for f in range(1,6):
                    vi=np.flatnonzero(fold==f);ti=np.flatnonzero(fold!=f)
                    tree,km,_=core.fit_strata_matrix(x0[ti],x0[vi],y0[ti],seed+f)
                    prepared.append((f,ti,vi,{'tree':tree,'kmeans':km},core.graphs(xy[vi])))
                all_rows=[];fold_rows=[];drows=[];replay_error=0.
                dataset={'coordinates':xy,'heterogeneity':h,'spatial_baseline':s0,'shared_uniforms':u,
                         'gaussian_noise':noise,'fold':fold,'generator_seed':seed}
                archived=pd.read_csv(OLD/'simulation/fold_components'/f'{key}.csv')
                for arm in ['fixed_amplitude']+(['variance_matched'] if scene in ACTIVE else []):
                    for lam in LOADINGS:
                        s=s0.copy() if lam==0 else std(np.sqrt(1-lam**2)*s0+lam*h)
                        diag,truth,y,phi_a,phi_h=diagnostics(hist,core,s0,s,h,a,b,prev,lam,arm,u,w,zone,seed)
                        if len(np.unique(y))<2:raise RuntimeError('One-class generated response; no replacement')
                        drows.append(dict(scenario=scene,repetition=rep,**diag))
                        tag=f'{arm}_l{lam:g}';dataset[tag+'_spatial']=s;dataset[tag+'_truth']=truth;dataset[tag+'_y']=y
                        x=np.c_[xy,s,h];adaptive=[];screens=[]
                        for f,ti,vi,frozen,gs in prepared:
                            if lam==0:labels=frozen
                            else:
                                tree,km,_=core.fit_strata_matrix(x[ti],x[vi],y[ti],seed+f)
                                labels={'tree':tree,'kmeans':km}
                            adaptive.append(labels)
                            # No post-screen exclusion: report ungated scores and diagnostic coverage separately.
                            screen=core.identifiability(y[vi],gs,frozen,199,seed*100+f)
                            screens.append(all(z['identified'] for z in screen))
                            for policy,levels in [('frozen',frozen),('adaptive',labels)]:
                                for name,z in levels.items():dataset[f'{tag}_{policy}_{name}_fold{f}']=z
                        for capture in CAPTURES:
                            raw=prev+capture*(truth-prev);p=np.clip(raw,core.P_CLIP,1-core.P_CLIP)
                            # Capture of each known probability contrast is exactly c before numerical bounds.
                            ga=(float((phi_a@(capture*phi_a))/(phi_a@phi_a)) if phi_a@phi_a>1e-20 else np.nan)
                            gh=(float((phi_h@(capture*phi_h))/(phi_h@phi_h)) if phi_h@phi_h>1e-20 else np.nan)
                            hist_result=hist.dsi(y,p,w,zone)
                            hc=dict(hist_eta_a=hist_result['eta_a'],hist_eta_h=hist_result['eta_h'],
                                    hist_theta=hist_result['theta_independent'])
                            parts={'frozen':[],'adaptive':[]}
                            for j,(f,ti,vi,frozen,gs) in enumerate(prepared):
                                frozen_values=core.theta_six(y[vi],p[vi],gs,frozen)
                                adaptive_values=with_levels(core,frozen_values,y[vi],p[vi],adaptive[j])
                                for policy,values in [('frozen',frozen_values),('adaptive',adaptive_values)]:
                                    parts[policy].append(values)
                                    fold_rows.append(dict(scenario=scene,repetition=rep,generator_seed=seed,arm=arm,loading=lam,
                                        capture_fraction=capture,strata_policy=policy,fold=f,validation_n=len(vi),**values))
                                if lam==0:
                                    old=archived.loc[archived['fold'].eq(f)&archived.capture_fraction.eq(capture)].iloc[0]
                                    for col in core.THETA+['r_dsi']:
                                        nv=frozen_values[col];ov=old[col]
                                        if np.isfinite(nv)!=np.isfinite(ov):raise RuntimeError('Baseline finite-status replay mismatch')
                                        if np.isfinite(nv):replay_error=max(replay_error,abs(nv-ov))
                            for policy,values in parts.items():
                                components=[f'eta_a_{r}' for r in core.RESIDUALS]+[f'eta_h_{z}_{r}' for r in core.RESIDUALS for z in core.STRATA]
                                agg={c:finite_mean([v[c] for v in values]) for c in core.THETA+components+['r_dsi']}
                                all_rows.append(dict(scenario=scene,repetition=rep,generator_seed=seed,arm=arm,loading=lam,
                                    capture_fraction=capture,strata_policy=policy,n=900,theta=agg['theta_tree_raw'],
                                    eta_a=agg['eta_a_raw'],eta_h=agg['eta_h_tree_raw'],**agg,**hc,
                                    finite_fold_fraction=float(np.mean([v['n_finite_theta']==6 for v in values])),
                                    frozen_response_identified_fold_fraction=float(np.mean(screens)),
                                    oracle_a_capture_gain=ga,oracle_h_capture_gain=gh,
                                    numerical_probability_bounds_n=int(np.sum(raw!=p)),
                                    maximum_probability_bound_change=float(np.max(abs(raw-p))),
                                    baseline_replay_max_error=replay_error))
                if replay_error>1e-7:raise RuntimeError(f'Baseline diagnostic replay mismatch {key}: {replay_error}')
                np.savez_compressed(BASE/'datasets'/f'{key}.npz',**dataset)
                csv(pd.DataFrame(fold_rows),BASE/'fold_components'/f'{key}.csv.gz')
                csv(pd.DataFrame(drows),BASE/'checkpoints'/f'{key}_diagnostics.csv')
                csv(pd.DataFrame(all_rows),checkpoint)
                if rep%10==0:logging.info('DONE %s baseline_error=%.3g elapsed=%.1fs',key,replay_error,time.perf_counter()-start)
    logging.info('Generation complete %.1fs; preparing summaries and final source verification',time.perf_counter()-start)

def rank_comparison(a,b,tol=1e-10):
    # Quantization is solely for the rank tie convention, never for stored scores.
    def tie_ranks(v):
        idx=np.argsort(v);rank=np.zeros(len(v));start=0
        while start<len(v):
            end=start+1
            while end<len(v) and abs(v[idx[end]]-v[idx[start]])<=tol:end+=1
            rank[idx[start:end]]=(start+end+1)/2;start=end
        return rank
    ra=tie_ranks(a);rb=tie_ranks(b);reverse=ties=0
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            sa=0 if abs(a[i]-a[j])<=tol else np.sign(a[i]-a[j]);sb=0 if abs(b[i]-b[j])<=tol else np.sign(b[i]-b[j])
            reverse+=int(sa*sb<0);ties+=int((sa==0)!=(sb==0))
    rho=float(spearmanr(ra,rb).statistic) if len(np.unique(ra))>1 and len(np.unique(rb))>1 else np.nan
    tau=float(kendalltau(ra,rb).statistic) if len(np.unique(ra))>1 and len(np.unique(rb))>1 else np.nan
    return rho,tau,reverse,ties

def summarize():
    files=sorted(p for p in (BASE/'checkpoints').glob('*_r*.csv') if not p.stem.endswith('_diagnostics'))
    raw=pd.concat([pd.read_csv(p,keep_default_na=False,na_values=['']) for p in files],ignore_index=True)
    diag=pd.concat([pd.read_csv(p) for p in (BASE/'checkpoints').glob('*_diagnostics.csv')],ignore_index=True)
    assert len(files)==600 and len(raw)==36000 and len(diag)==3600,(len(files),len(raw),len(diag))
    fold_n=sum(len(pd.read_csv(p,usecols=['fold'])) for p in (BASE/'fold_components').glob('*.csv.gz'))
    assert fold_n==180000,fold_n
    csv(raw,BASE/'repetition_results.csv.gz');csv(diag,BASE/'generator_diagnostics.csv')
    metrics=['eta_a','eta_h','theta','r_dsi','hist_eta_a','hist_eta_h','hist_theta']
    keys=['scenario','arm','strata_policy','capture_fraction']
    summary=[];paired=[];delta_raw=[];ranks=[]
    rng=np.random.default_rng(BOOT_SEED)
    for group,part in raw.groupby(keys,sort=False):
        base=dict(zip(keys,group));reps=np.sort(part.repetition.unique());indices=rng.integers(0,len(reps),(BOOTSTRAPS,len(reps)))
        for metric in metrics:
            mat=part.pivot(index='repetition',columns='loading',values=metric).reindex(index=reps,columns=LOADINGS).to_numpy()
            for j,lam in enumerate(LOADINGS):
                vals=mat[:,j];finite=vals[np.isfinite(vals)]
                def stats(v,bootstrap):
                    v=v[np.isfinite(v)]
                    if not len(v):return dict(n_finite=0,n_missing=100,mean=np.nan,median=np.nan,q1=np.nan,q3=np.nan,iqr=np.nan,min=np.nan,max=np.nan,lower_2p5=np.nan,upper_97p5=np.nan,mean_ci_lower=np.nan,mean_ci_upper=np.nan,median_ci_lower=np.nan,median_ci_upper=np.nan,below_zero=0,above_one=0)
                    samples=bootstrap
                    with np.errstate(invalid='ignore'):
                        bm=np.nanmean(samples,axis=1);bd=np.nanmedian(samples,axis=1)
                    return dict(n_finite=len(v),n_missing=100-len(v),mean=float(v.mean()),median=float(np.median(v)),q1=float(np.quantile(v,.25)),q3=float(np.quantile(v,.75)),iqr=float(np.quantile(v,.75)-np.quantile(v,.25)),min=float(v.min()),max=float(v.max()),lower_2p5=float(np.quantile(v,.025)),upper_97p5=float(np.quantile(v,.975)),mean_ci_lower=float(np.nanquantile(bm,.025)),mean_ci_upper=float(np.nanquantile(bm,.975)),median_ci_lower=float(np.nanquantile(bd,.025)),median_ci_upper=float(np.nanquantile(bd,.975)),below_zero=int(np.sum(v<0)),above_one=int(np.sum(v>1)))
                summary.append(dict(**base,loading=lam,metric=metric,**stats(vals,vals[indices])))
                dv=mat[:,j]-mat[:,0];dv[~(np.isfinite(mat[:,j])&np.isfinite(mat[:,0]))]=np.nan
                paired.append(dict(**base,loading=lam,metric='delta_'+metric,**stats(dv,dv[indices])))
                for rep,value in zip(reps,dv):delta_raw.append(dict(**base,loading=lam,repetition=int(rep),metric=metric,paired_delta=value))
    summary=pd.DataFrame(summary);paired=pd.DataFrame(paired)
    csv(summary,BASE/'tables/distribution_summary.csv');csv(paired,BASE/'tables/paired_delta_summary.csv')
    csv(pd.DataFrame(delta_raw),BASE/'paired_deltas.csv.gz')
    diagnostic_summary=[]
    dm=['driver_correlation','spatial_driver_moran','spatial_driver_q','response_moran','response_q_fixed_bands',
        'observed_prevalence','expected_prevalence','signal_variance','effective_a','effective_h','common_amplitude_multiplier']
    for group,part in diag.groupby(['scenario','arm','loading'],sort=False):
        for m in dm:
            v=part[m].to_numpy();diagnostic_summary.append(dict(scenario=group[0],arm=group[1],loading=group[2],metric=m,n=len(v),median=float(np.median(v)),q1=float(np.quantile(v,.25)),q3=float(np.quantile(v,.75)),min=float(v.min()),max=float(v.max())))
    csv(pd.DataFrame(diagnostic_summary),BASE/'tables/generator_diagnostic_summary.csv')
    for group,part in raw.groupby(['scenario','arm','strata_policy'],sort=False):
        for metric in ['theta','r_dsi','hist_theta']:
            p=part.pivot(index='repetition',columns=['loading','capture_fraction'],values=metric)
            cols=pd.MultiIndex.from_product([LOADINGS,CAPTURES]);p=p.reindex(columns=cols)
            complete=p.loc[np.isfinite(p.to_numpy()).all(axis=1)]
            for lam in LOADINGS:
                scores0=complete[0.].median().to_numpy();scores1=complete[lam].median().to_numpy()
                rho,tau,reversal,ties=rank_comparison(scores0,scores1) if len(complete) else (np.nan,np.nan,0,0)
                ranks.append(dict(scenario=group[0],arm=group[1],strata_policy=group[2],metric=metric,loading=lam,complete_paired_repetitions=len(complete),configuration_n=5,pair_n=10,spearman_rho=rho,kendall_tau_b=tau,reversed_pairs=reversal,tie_status_changes=ties))
    csv(pd.DataFrame(ranks),BASE/'tables/capture_configuration_ranking.csv')
    # Algebraic attribution of historical paired changes, not a true explanation target.
    decomp=[]
    for group,part in raw.query("strata_policy == 'frozen'").groupby(['scenario','arm','capture_fraction']):
        b=part.query('loading == 0').set_index('repetition')
        for lam in LOADINGS:
            p=part.loc[part.loading.eq(lam)].set_index('repetition').loc[b.index]
            da=p.hist_eta_a-b.hist_eta_a;dh=p.hist_eta_h-b.hist_eta_h
            terms=[(1-b.hist_eta_h)*da,(1-b.hist_eta_a)*dh,-da*dh]
            error=p.hist_theta-b.hist_theta-sum(terms)
            for i,rep in enumerate(b.index):decomp.append(dict(scenario=group[0],arm=group[1],capture_fraction=group[2],loading=lam,repetition=rep,delta_theta=float((p.hist_theta-b.hist_theta).iloc[i]),autocorrelation_term=float(terms[0].iloc[i]),heterogeneity_term=float(terms[1].iloc[i]),interaction_term=float(terms[2].iloc[i]),algebra_error=float(error.iloc[i])))
    csv(pd.DataFrame(decomp),BASE/'tables/historical_delta_algebra.csv.gz')
    gradient=diag.query("arm == 'fixed_amplitude'").pivot(index=['scenario','repetition'],columns='loading',values='driver_correlation')
    assert np.all(np.diff(gradient.to_numpy(),axis=1)>0)
    assert np.max(abs(diag.spatial_sd-1))<1e-12 and np.max(abs(diag.heterogeneity_sd-1))<1e-12
    assert np.max(abs(diag.expected_prevalence-diag.target_prevalence))<1e-12
    vm=diag.query("arm == 'variance_matched'")
    assert np.max(abs(vm.signal_variance-vm.baseline_signal_variance))<1e-10
    gains=raw[['oracle_a_capture_gain','oracle_h_capture_gain']].to_numpy()-raw.capture_fraction.to_numpy()[:,None]
    assert np.nanmax(abs(gains))<1e-12
    neg=raw.query('capture_fraction == 0')
    assert np.nanmax(abs(neg[['theta','hist_theta','r_dsi']].to_numpy()))<1e-7
    audit=dict(success=True,independent_seed_blocks=600,condition_datasets=len(diag),repetition_configuration_rows=len(raw),
        expected_repetition_configuration_rows=36000,fold_component_rows=fold_n,
        baseline_maximum_diagnostic_replay_error=float(raw.baseline_replay_max_error.max()),
        missing_by_metric={m:int((~np.isfinite(raw[m])).sum()) for m in metrics},
        nonfinite_infinite_by_metric={m:int(np.isinf(raw[m]).sum()) for m in metrics},
        maximum_oracle_capture_gain_error=float(np.nanmax(abs(gains))),
        probability_bound_rows=int(raw.numerical_probability_bounds_n.gt(0).sum()),
        maximum_probability_bound_change=float(raw.maximum_probability_bound_change.max()),
        dependency_gradient_monotone_all_seed_blocks=True,
        source_files_verified_unchanged=check_sources(),
        primary_identifiability='ungated finite scores; frozen-strata response screen fractions reported separately',
        conditioning='finite distribution values; paired contrasts require both values finite; ranks use all-loading/all-capture complete repetitions')
    write_json(BASE/'qa/completion.json',audit)
    logging.info('SUMMARY COMPLETE rows=%d; missing=%s',len(raw),audit['missing_by_metric'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot-only',action='store_true');parser.add_argument('--summarize-only',action='store_true');args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(BASE/'run.log',encoding='utf-8')])
    core,hist=prepare();logging.info('Command %s',sys.argv)
    if not args.summarize_only:run(core,hist,args.pilot_only)
    if not args.pilot_only:summarize()

if __name__=='__main__':
    try:main()
    except Exception:
        logging.exception('FAILED; no missing scenarios or results fabricated');raise
