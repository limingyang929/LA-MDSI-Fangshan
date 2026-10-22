"""Replay the inherited binary DGP; explicitly new six-setting diagnostic extension."""
from scientific_core import *
import argparse
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import norm
from threadpoolctl import threadpool_limits

def sigmoid(z):return 1/(1+np.exp(-np.clip(z,-30,30)))

def intercept(signal,target):
    lo,hi=-20.0,20.0
    for _ in range(80):
        mid=(lo+hi)/2
        if sigmoid(mid+signal).mean()<target:lo=mid
        else:hi=mid
    return (lo+hi)/2

def finite_mean(values):
    a=np.asarray(values,float)
    return float(a.mean()) if np.isfinite(a).all() else np.nan

def historical_score(y,p,w,z):
    """Independent direct implementation of the existing single-setting simulation equations."""
    n=len(y);v=y-p;iy=moran(y,w);ir=moran(v,w);qy=q_stat(y,z);qr=q_stat(v,z);eq=(len(np.unique(z))-1)/(n-1)
    ao=eta(iy,ir);ho=eta(qy,qr);a=eta(iy,ir,-1/(n-1));h=eta(qy,qr,eq,1-eq)
    return dict(historical_i_y=iy,historical_q_y=qy,historical_eta_a=a,historical_eta_h=h,historical_theta_original_abs=joint(ao,ho),historical_theta_centered=joint(a,h))

def summarize(results):
    metrics=['dsi_original_theta3','theta','r_dsi','s_theta','r_dsi_gated','roc_auc','brier','pr_auc','log_loss','ece10']+THETA+['historical_theta_original_abs','historical_theta_centered']
    rows=[];intervals=[]
    rng=np.random.default_rng(20260824)
    for (scenario,capture),frame in results.groupby(['scenario','capture_fraction'],sort=False):
        for metric in metrics:
            values=frame[metric].to_numpy(float);finite=values[np.isfinite(values)]
            row=dict(scenario=scenario,capture_fraction=capture,metric=metric,n_repetitions=len(values),n_finite=len(finite),n_undefined=int((~np.isfinite(values)).sum()),mean=float(finite.mean()) if len(finite) else np.nan,median=float(np.median(finite)) if len(finite) else np.nan,sd=float(finite.std(ddof=1)) if len(finite)>1 else np.nan,lower_2p5=float(np.quantile(finite,.025)) if len(finite) else np.nan,upper_97p5=float(np.quantile(finite,.975)) if len(finite) else np.nan,interval_kind='central 95% distribution over actual independent DGP repetitions; not CI of mean')
            rows.append(row)
            if len(finite)>1:
                means=finite[rng.integers(0,len(finite),size=(10000,len(finite)))].mean(axis=1)
                intervals.append({**row,'MC_mean_CI_lower':float(np.quantile(means,.025)),'MC_mean_CI_upper':float(np.quantile(means,.975)),'MC_bootstrap_repetitions':10000,'MC_bootstrap_seed':20260824,'conditioning':'only mathematically defined values; undefined fraction reported, no zero imputation'})
            else:intervals.append({**row,'MC_mean_CI_lower':np.nan,'MC_mean_CI_upper':np.nan,'MC_bootstrap_repetitions':0,'MC_bootstrap_seed':20260824,'conditioning':'no estimable interval'})
    pd.DataFrame(rows).to_csv(BASE/'simulation_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(intervals).to_csv(BASE/'simulation_intervals.csv',index=False,encoding='utf-8-sig')
    control=results.loc[results.scenario.eq('null')|results.capture_fraction.eq(0)].copy()
    control['expected_theta_zero']=True
    control['max_abs_six_theta']=np.abs(control[THETA].to_numpy(float)).max(axis=1)
    control['zero_effect_within_1e_8']=control['max_abs_six_theta'].le(1e-8)
    control['zero_effect_status']=np.where(control[THETA].notna().all(axis=1),'numerically evaluable','undefined CART/q or spatial ratio; not replaced by zero')
    control.to_csv(BASE/'simulation_negative_controls.csv',index=False,encoding='utf-8-sig')
    coverage=results.drop_duplicates(['scenario','repetition']).groupby('scenario',as_index=False).agg(repetitions=('repetition','size'),prevalence_interval_coverage=('prevalence_interval_covers_target','mean'),mean_response_identified_fold_fraction=('response_identified_fold_fraction','mean'),mean_finite_fold_fraction=('finite_fold_fraction','mean'))
    coverage['coverage_estimand']='known mean conditional Bernoulli prevalence, not theta/R-DSI'
    coverage.to_csv(BASE/'simulation_stability_coverage.csv',index=False,encoding='utf-8-sig')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    log=logger('simulation');begin=time.perf_counter()
    cfg=json.loads((CONFIG/'02_Legacy_simulation.json').read_text(encoding='utf-8'));old=cfg['historical_replay']
    out=BASE/'simulation';out.mkdir(exist_ok=True)
    snapshot=out/'frozen_protocol.json';cfgsha=sha(CONFIG/'02_Legacy_simulation.json')
    if snapshot.exists():assert json.loads(snapshot.read_text(encoding='utf-8'))['config_sha256']==cfgsha
    else:write_json(snapshot,dict(fixed_before_first_generation_utc=datetime.now(timezone.utc).isoformat(),config_sha256=cfgsha,runner_sha256=sha(Path(__file__)),scientific_core_sha256=sha(CODE/'scientific_core.py')))
    side=old['grid_side'];gy,gx=np.mgrid[0:side,0:side];xy=np.c_[gx.ravel(),gy.ravel()]
    zone=np.digitize(gx.ravel(),[6,12,18,24]);hy=np.array([-1.3,.7,-.5,1.4,-.3])[zone];hy=(hy-hy.mean())/hy.std()
    w=graph(xy,8,'legacy');blocks=block_ids(xy,6)
    checkpoints=out/'checkpoints';checkpoints.mkdir(exist_ok=True)
    datasets=out/'datasets';datasets.mkdir(exist_ok=True)
    fold_dir=out/'fold_components';fold_dir.mkdir(exist_ok=True)
    repetitions=1 if args.smoke else 100
    with threadpool_limits(limits=1):
        for scenario_index,(scenario,(a,h,prev)) in enumerate(old['scenarios'].items()):
            for rep in range(1,repetitions+1):
                key=f'{scenario}_r{rep:03d}';target=checkpoints/f'{key}.csv'
                if target.exists():continue
                seed=20260824+scenario_index*10000+rep;rng=np.random.default_rng(seed)
                noise=rng.normal(size=(side,side));spatial=gaussian_filter(noise,sigma=3.0,mode='reflect').ravel();spatial=(spatial-spatial.mean())/spatial.std()
                signal=a*spatial+h*hy;true_p=sigmoid(intercept(signal,prev)+signal);y=rng.binomial(1,true_p)
                if len(np.unique(y))<2:raise RuntimeError(f'{key}: response has only one class; no invented replacement')
                np.savez_compressed(datasets/f'{key}.npz',coordinates=xy,response=y,true_probability=true_p,spatial=spatial,heterogeneity=hy,gaussian_noise=noise,generator_seed=seed)
                # Replay the original null permutations before capture predictions.
                null_i=[];null_q=[]
                for _ in range(199):
                    yp=rng.permutation(y);null_i.append(moran(yp,w));null_q.append(q_stat(yp,zone))
                def pvalue(observed,null):
                    z=np.asarray(null);return float((1+np.sum(abs(z-z.mean())>=abs(observed-z.mean())))/(len(z)+1))
                pi=pvalue(moran(y,w),null_i);pq=pvalue(q_stat(y,zone),null_q)
                x=np.c_[xy,spatial,hy]
                splits=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed).split(x,y,blocks))
                folded=np.zeros(len(y),int);prepared=[]
                for fold,(ti,vi) in enumerate(splits,1):
                    assert set(blocks[ti]).isdisjoint(blocks[vi]);folded[vi]=fold
                    tl,km,rules=fit_strata_matrix(x[ti],x[vi],y[ti],seed+fold)
                    levels={'tree':tl,'kmeans':km};gs=graphs(xy[vi]);og=graph(xy[vi],8,'original')
                    screen=identifiability(y[vi],gs,levels,199,seed*100+fold)
                    prepared.append((fold,vi,levels,gs,og,all(r['identified'] for r in screen)))
                    sd=out/'screens';sd.mkdir(exist_ok=True);pd.DataFrame(screen).to_csv(sd/f'{key}_f{fold}.csv',index=False)
                np.savez_compressed(out/f'{key}_fold_assignment.npz',fold=folded,blocks=blocks)
                rep_rows=[];fold_rows=[]
                for capture in old['capture_fractions']:
                    p=np.clip(prev+capture*(true_p-prev),P_CLIP,1-P_CLIP);parts=[];original=[]
                    for fold,vi,levels,gs,og,identified in prepared:
                        values=theta_six(y[vi],p[vi],gs,levels)
                        rows,_=ablation(y[vi],p[vi],og,gs,levels)
                        orig=rows[0]['dsi_theta3'];original.append(orig);parts.append(values)
                        fold_rows.append(dict(scenario=scenario,repetition=rep,generator_seed=seed,fold=fold,capture_fraction=capture,validation_n=len(vi),response_identifiable=identified,dsi_original_theta3=orig,**values))
                    finite=[v['n_finite_theta']==6 for v in parts];identified=[v[-1] for v in prepared]
                    six_mean={c:finite_mean([d[c] for d in parts]) for c in THETA}
                    sd_known=np.sqrt(np.sum(true_p*(1-true_p)))/len(y);margin=norm.ppf(.975)*sd_known
                    rep_rows.append(dict(scenario=scenario,repetition=rep,generator_seed=seed,spatial_strength=a,heterogeneity_strength=h,target_prevalence=prev,observed_prevalence=float(y.mean()),capture_fraction=capture,n=len(y),prediction_type='controlled_oracle_no_validation_label_fit',LA_extension_protocol='new_design_train_only_5fold_6cell_blocks',dsi_original_theta3=finite_mean(original),theta=six_mean['theta_tree_raw'],**six_mean,r_dsi=finite_mean([v['r_dsi'] for v in parts]),s_theta=finite_mean([v['s_theta'] for v in parts]),r_dsi_gated=finite_mean([v['r_dsi'] for v in parts]) if all(identified) else np.nan,response_identified_fold_fraction=float(np.mean(identified)),finite_fold_fraction=float(np.mean(finite)),undefined_reason='' if all(finite) else 'one_or_more_fold_CART/q_or_ratio_undefined',historical_moran_permutation_p=pi,historical_q_permutation_p=pq,prevalence_interval_lower=float(y.mean()-margin),prevalence_interval_upper=float(y.mean()+margin),prevalence_interval_covers_target=bool(y.mean()-margin<=prev<=y.mean()+margin),**predictive(y,p),**historical_score(y,p,w,zone)))
                pd.DataFrame(fold_rows).to_csv(fold_dir/f'{key}.csv',index=False)
                pd.DataFrame(rep_rows).to_csv(target,index=False)
                log.info('DONE %s seed=%d finite_fold_fraction=%.2f identified_fold_fraction=%.2f elapsed=%.2fs',key,seed,np.mean(finite),np.mean(identified),time.perf_counter()-begin)
    # "null" is a literal scenario label, not a missing-value token.
    paths=sorted(checkpoints.glob('*.csv'));result=pd.concat([pd.read_csv(p,float_precision='round_trip',keep_default_na=False,na_values=['']) for p in paths],ignore_index=True)
    expected=30 if args.smoke else 3000
    assert len(result)==expected,(len(result),expected)
    result.to_csv(BASE/'simulation_raw_results.csv',index=False,encoding='utf-8-sig');summarize(result)
    assert result.scenario.nunique()==6
    write_json(out/('smoke_completion.json' if args.smoke else 'completion.json'),dict(success=True,scenario_count=6,expected_captures=expected,completed_captures=len(result),independent_DGP_repetitions=len(paths),wall_seconds=time.perf_counter()-begin,config_sha256=cfgsha,aggregation_runner_sha256=sha(Path(__file__)),aggregation_command=subprocess.list2cmdline([sys.executable,*sys.argv]),utc=datetime.now(timezone.utc).isoformat()))
    log.info('COMPLETE captures=%d independent_DGPs=%d wall=%.3fs',len(result),len(paths),time.perf_counter()-begin)

if __name__=='__main__':
    try:main()
    except Exception:
        logging.getLogger('simulation').exception('FAILED; missing repetitions are not fabricated')
        raise
