"""Paired extension and boundary controls on the frozen 69-cell simulation.

Original observations, code, graphs, labels and manuscript files are read-only.
No outcome-based selection of worlds, sizes, sampling arms or thresholds.
"""
from __future__ import annotations
import os
import sys
for task_thread_option in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[task_thread_option] = '1'
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

from project_config import *
OUT = ensure_output(REPLACEMENT)
SOURCE = OUTPUT_BASE / '01_Full/03_Sampling_simulation'
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run_experiment as run
EXT = OUT / 'extended_sizes'
ALPHAS = [.025, .05, .10, .20]
KEY = ['world_id','N','arm','fold_count']


def now(): return datetime.now(timezone.utc).isoformat()
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''): h.update(part)
    return h.hexdigest()
def dump(name, obj):
    path=(OUT/name).resolve()
    assert path.is_relative_to(OUT.resolve())
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(name, frame):
    path=(OUT/name).resolve()
    assert path.is_relative_to(OUT.resolve())
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,encoding='utf-8-sig')
def read(path):
    return pd.read_csv(path,keep_default_na=False,na_values=[''],float_precision='round_trip',low_memory=False)
def truth(series):
    # Missing values stringify to a non-true token; no downcasting or zero imputation.
    return series.astype(str).str.lower().eq('true')


def freeze():
    path=OUT/'replacement_protocol.json'
    protected=[SOURCE/p for p in ('protocol.json','geometry.npz','datasets.csv','conditions.csv','folds.csv','screens.csv','graphs.csv','samples.csv')]
    protected += [CODE/'run_experiment.py',run.REFERENCE_CORE,run.REFERENCE_GRAPH]
    manifest={str(p):digest(p) for p in protected}
    if path.exists():
        saved=json.loads(path.read_text(encoding='utf-8'))
        assert saved['protected_inputs']==manifest, 'Frozen source inputs changed'
        return saved
    protocol=dict(frozen_utc=now(),purpose='Evaluate replacement of the original 25-group N900 simulation by one coherent 69-cell simulation family',
        status='Additional analyses requested after prior results; not preregistered and not independent external validation',
        original_worlds=600,independent_worlds_per_scenario=100,
        original_sample_sizes=[900,3600],additional_sample_sizes=[400,8100],
        original_fold_counts=[5,3],additional_size_fold_counts=[5],
        sampling_arms=list(run.ARMS),capture_conditions=list(run.CAPTURES),
        main_alpha=.05,threshold_sensitivity=ALPHAS,threshold_rule='strict p < alpha',
        permutations=499,holm='Adjust six response tests within each fold; show separately',
        primary_gate='All six valid response tests in all five folds, plus complete finite scores',
        formula_and_parameters='Original eta ratios, product-complement theta, Type-7 lower quartile, 1-IQR, epsilon1e-12, probability clipping1e-6, negative scores retained',
        new_size_generation='Reuse frozen runner, seeds, 110400 candidates, signal parameters and sampling priorities; newly assign five spatial folds at each sample size; no duplication, redrawing or quota relaxation',
        boundary_control='p_i=1-epsilon when observed y_i=1, otherwise epsilon; constructed using labels to test metric properties, not predictive validation',
        boundary_scope='All four sample sizes, all six scenarios and all four sampling arms, five folds, whenever sampling was feasible',
        boundary_expected='Every residual definition is affine in binary response. Moran I and q invariant to nonzero affine scaling, so eta/theta/RDSI=0 and S_theta=1 where defined. Numerical failures retained.',
        fail_policy='All requested worlds stay in denominators; invalid/missing screens fail eligibility; no failed folds or settings omitted',
        scope_not_changed=['Fangshan models','empirical results','formulas','main alpha','five-fold primary rule','original files','Word manuscripts'],
        protected_inputs=manifest,initial_runner_sha256=digest(__file__),python=sys.executable)
    dump('replacement_protocol.json',protocol)
    EXT.mkdir(exist_ok=True)
    shutil.copyfile(SOURCE/'geometry.npz',EXT/'geometry.npz')
    return protocol


def extension_worker(task):
    index,repetition=task
    run.OUT=EXT
    run.SIZES=(400,8100)
    run.FOLDS=(5,)
    result=run.world_task(index,repetition)
    key=f'{list(run.SCENARIOS)[index]}_r{repetition:03d}'
    old=json.loads((SOURCE/'worlds'/key/'completion.json').read_text(encoding='utf-8'))
    new=json.loads((EXT/'worlds'/key/'completion.json').read_text(encoding='utf-8'))
    assert old['population_sha256']==new['population_sha256'], key
    return {**result,'population_identical':True}


def extend(workers=8,smoke=False):
    freeze()
    tasks=[(i,r) for i in range(6) for r in range(1,101)]
    if smoke: tasks=[(0,1),(5,1)]
    started=time.perf_counter()
    checks=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(extension_worker,t):t for t in tasks}
        for f in as_completed(pending):
            checks.append(f.result())
            done=len(checks)
            if done%20==0 or done==len(tasks):
                dump('extension_progress.json',dict(completed=done,requested=len(tasks),seconds=time.perf_counter()-started,last=checks[-1],utc=now()))
                print(f'EXTENSION {done}/{len(tasks)} in {time.perf_counter()-started:.1f}s',flush=True)
    dump('extension_smoke.json' if smoke else 'extension_completion.json',dict(success=True,worlds=len(checks),all_populations_identical=True,seconds=time.perf_counter()-started,results=checks))
    if not smoke:
        for name in ('samples','datasets','conditions','folds','screens','graphs'):
            frames=[read(p) for p in sorted((EXT/'worlds').glob(f'*/{name}.csv'))]
            csv(f'extended_sizes/{name}.csv',pd.concat(frames,ignore_index=True))
        print('EXTENSION MERGED',flush=True)


def boundary_worker(task):
    key=task
    target=OUT/'boundary_worlds'/key
    target.mkdir(parents=True,exist_ok=True)
    if (target/'complete.json').exists(): return dict(key=key,cached=True)
    geom=np.load(SOURCE/'geometry.npz',allow_pickle=False)
    xy=geom['coordinates']
    rows=[];summaries=[];checks=[]
    with threadpool_limits(limits=1):
        for base,sizes in [(SOURCE,(900,3600)),(EXT,(400,8100))]:
            root=base/'worlds'/key
            dataset=read(root/'datasets.csv')
            fold_source=read(root/'folds.csv')
            conditions=read(root/'conditions.csv')
            for n in sizes:
                for arm in run.ARMS:
                    meta=dataset[(dataset.N==n)&(dataset.arm==arm)&(dataset.fold_count==5)].iloc[0]
                    common={name:meta[name] for name in ['world_id','scenario','N','arm','fold_count']}
                    if not truth(pd.Series([meta.sampling_feasible])).iloc[0]:
                        summaries.append({**common,'sampling_feasible':False,'numeric_finite':False,'response_all_folds_pass':False,'fully_reportable':False,'roc_auc':np.nan,'brier':np.nan,'r_dsi':np.nan,'s_theta':np.nan,'failure_reason':'sampling_infeasible'})
                        continue
                    sample=np.load(root/f'sample_N{n}_{arm}.npz',allow_pickle=False)
                    stored=np.load(root/f'folds_N{n}_{arm}_cv5.npz',allow_pickle=False)
                    y=sample['response'];indices=sample['candidate_indices']
                    probability=np.where(y==1,1-run.sc.P_CLIP,run.sc.P_CLIP)
                    parts=[]
                    for fold in range(1,6):
                        vi=np.flatnonzero(stored['assignment']==fold)
                        levels={'tree':stored['tree_labels'][vi],'kmeans':stored['kmeans_labels'][vi]}
                        graphs,audits=run.correct_graphs(xy[indices[vi]])
                        with np.errstate(divide='ignore',invalid='ignore'):
                            val=run.diagnostics(y[vi],probability[vi],graphs,levels)
                        finite=val['n_finite_theta']==6
                        reason=run.failure_reason(y[vi],levels,val)
                        source_row=fold_source[(fold_source.N==n)&(fold_source.arm==arm)&(fold_source.fold_count==5)&(fold_source.fold==fold)].iloc[0]
                        expected_sha=str(source_row.validation_index_sha256)
                        assert hashlib.sha256(indices[vi].tobytes()).hexdigest()==expected_sha
                        parts.append(val)
                        rows.append({**common,'fold':fold,'validation_n':len(vi),'validation_positive':int(y[vi].sum()),'numeric_finite':finite,'failure_reason':reason,
                            'response_six_tests_pass':bool(source_row.six_tests_pass),'r_dsi':val['r_dsi'],'s_theta':val['s_theta'],
                            **{name:val[name] for name in run.sc.THETA},
                            **{name:val[name] for name in val if name.startswith(('eta_a_','eta_h_'))},
                            'validation_index_sha256':expected_sha})
                    numeric=all(v['n_finite_theta']==6 for v in parts)
                    summaries.append({**common,'sampling_feasible':True,'numeric_finite':numeric,
                        'response_all_folds_pass':bool(meta.response_all_folds_pass),
                        'fully_reportable':bool(numeric and meta.response_all_folds_pass),
                        'roc_auc':float(run.sc.roc_auc_score(y,probability)) if np.unique(y).size==2 else np.nan,
                        'brier':float(np.mean((y-probability)**2)),
                        'r_dsi':run.complete_mean([v['r_dsi'] for v in parts]),
                        's_theta':run.complete_mean([v['s_theta'] for v in parts]),
                        'failure_reason':'' if numeric else ';'.join(sorted(set(r['failure_reason'] for r in rows if r['N']==n and r['arm']==arm and r['failure_reason'])))})
                    zero=conditions[(conditions.N==n)&(conditions.arm==arm)&(conditions.fold_count==5)&(conditions.capture==0)].iloc[0]
                    checks.append({**common,'zero_capture_rdsi':zero.r_dsi,'zero_capture_finite':bool(zero.numeric_finite),
                        'boundary_rdsi':summaries[-1]['r_dsi'],'boundary_finite':numeric})
    for name,table in [('folds',rows),('datasets',summaries),('paired_constant',checks)]:
        pd.DataFrame(table).to_csv(target/f'{name}.csv',index=False,encoding='utf-8-sig')
    (target/'complete.json').write_text(json.dumps(dict(success=True,key=key,dataset_controls=len(summaries),fold_controls=len(rows)),indent=2),encoding='utf-8')
    return dict(key=key,rows=len(rows))


def boundary(workers=8,smoke=False,ready_only=False):
    freeze()
    tasks=[f'{scenario}_r{r:03d}' for scenario in run.SCENARIOS for r in range(1,101)]
    if smoke: tasks=['null_r001','mixed_rare_r001']
    if ready_only:
        tasks=[t for t in tasks if (EXT/'worlds'/t/'completion.json').exists() and not (OUT/'boundary_worlds'/t/'complete.json').exists()]
    started=time.perf_counter();completed=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(boundary_worker,t) for t in tasks]
        for f in as_completed(futures):
            result=f.result();completed+=1
            if completed%30==0 or completed==len(tasks):
                dump('boundary_progress.json',dict(completed=completed,requested=len(tasks),seconds=time.perf_counter()-started,utc=now()))
                print(f'BOUNDARY {completed}/{len(tasks)} in {time.perf_counter()-started:.1f}s',flush=True)
    if not smoke and not ready_only:
        for name in ('folds','datasets','paired_constant'):
            csv(f'boundary_{name}.csv',pd.concat([read(p) for p in sorted((OUT/'boundary_worlds').glob(f'*/{name}.csv'))],ignore_index=True))
    name='boundary_smoke.json' if smoke else ('boundary_batch.json' if ready_only else 'boundary_completion.json')
    dump(name,dict(success=True,worlds=completed,seconds=time.perf_counter()-started,utc=now()))


def summarize():
    freeze()
    tables={}
    for name in ('datasets','conditions','screens','folds','graphs','samples'):
        tables[name]=pd.concat([read(SOURCE/f'{name}.csv'),read(EXT/f'{name}.csv')],ignore_index=True)
    data,cond,screen,fold=tables['datasets'],tables['conditions'],tables['screens'],tables['folds']
    assert len(data)==14400
    assert not data.duplicated(KEY).any()
    assert not cond.duplicated(KEY+['capture']).any()
    assert data.world_id.nunique()==600
    for col in ('numeric_at_capture1','reportable_at_capture1','sampling_feasible','valid_all_fold_screens'):
        data[col]=truth(data[col])
    assert len(cond)==5*int(data.sampling_feasible.sum())
    assert len(fold)==int(data.loc[data.sampling_feasible,'fold_count'].sum())
    assert len(screen)==6*len(fold)
    cond['numeric_finite']=truth(cond.numeric_finite)
    cond['fully_reportable']=truth(cond.fully_reportable)
    screen['valid']=truth(screen.screen_valid)&np.isfinite(screen.permutation_p)
    sf=screen.groupby(KEY+['fold'],dropna=False).agg(count=('setting','size'),unique=('setting','nunique'),valid=('valid','all'),max_p=('permutation_p','max'),max_holm=('holm_p','max')).reset_index()
    sf['valid']=sf['valid']&sf['count'].eq(6)&sf['unique'].eq(6)
    eligibility=[];summary=[]
    for alpha in ALPHAS:
        for adjustment,field in [('nominal','max_p'),('holm','max_holm')]:
            sf['pass']=sf.valid&sf[field].lt(alpha)
            ds=sf.groupby(KEY,dropna=False).agg(fold_rows=('fold','size'),valid_folds=('valid','sum'),passing_folds=('pass','sum')).reset_index()
            merged=data.merge(ds,on=KEY,how='left',validate='one_to_one',suffixes=('','_recomputed'))
            # Original metadata also contain valid_folds/passing_folds; recomputed versions are authoritative here.
            for col in ('fold_rows','valid_folds_recomputed','passing_folds_recomputed'):
                if col not in merged:
                    plain=col.replace('_recomputed','');merged[col]=merged[plain]
                merged[col]=merged[col].fillna(0).astype(int)
            merged['eligible']=merged.sampling_feasible&merged.numeric_at_capture1&merged.fold_rows.eq(merged.fold_count)&merged.valid_folds_recomputed.eq(merged.fold_count)&merged.passing_folds_recomputed.eq(merged.fold_count)
            merged['eligible_four_of_five']=merged.sampling_feasible&merged.numeric_at_capture1&merged.fold_count.eq(5)&merged.fold_rows.eq(5)&merged.passing_folds_recomputed.ge(4)
            if alpha==.05 and adjustment=='nominal':
                assert np.array_equal(merged.eligible.to_numpy(),merged.reportable_at_capture1.to_numpy()),'Main counts failed reconciliation'
            if alpha==.05:
                eligibility.append(merged[KEY+['scenario','eligible','eligible_four_of_five']].assign(adjustment=adjustment))
            for group,g in merged.groupby(['N','arm','fold_count','scenario'],sort=False):
                assert len(g)==100
                n,arm,cv,scene=group
                summary.append(dict(N=int(n),arm=arm,fold_count=int(cv),scenario=scene,alpha=alpha,adjustment=adjustment,
                    worlds=100,sampling_feasible=int(g.sampling_feasible.sum()),numeric_finite=int(g.numeric_at_capture1.sum()),
                    valid_all_screens=int((g.valid_folds_recomputed==cv).sum()),fully_reportable=int(g.eligible.sum()),rate=float(g.eligible.mean()),
                    four_of_five=int(g.eligible_four_of_five.sum()) if cv==5 else np.nan))
    threshold=pd.DataFrame(summary)
    csv('threshold_sensitivity_all.csv',threshold)
    csv('eligibility_at_main_alpha.csv',pd.concat(eligibility,ignore_index=True))
    main=threshold[(threshold.alpha==.05)&(threshold.adjustment=='nominal')&(threshold.fold_count==5)]
    csv('main_reportability_all_sizes_arms.csv',main)
    totals=main.groupby(['N','arm']).agg(worlds=('worlds','sum'),sampling_feasible=('sampling_feasible','sum'),numeric_finite=('numeric_finite','sum'),fully_reportable=('fully_reportable','sum')).reset_index()
    totals['rate']=totals.fully_reportable/totals.worlds
    csv('main_totals.csv',totals)
    boundary=read(OUT/'boundary_datasets.csv')
    for col in ('sampling_feasible','numeric_finite','fully_reportable'):boundary[col]=truth(boundary[col])
    bsummary=[]
    for group,g in boundary.groupby(['N','arm','scenario']):
        finite=g[truth(g.numeric_finite)]
        assert len(g)==100
        n,arm,scene=group
        bsummary.append(dict(N=int(n),arm=arm,scenario=scene,worlds=100,sampling_feasible=int(g.sampling_feasible.sum()),
            numerical_finite=len(finite),fully_reportable=int(g.fully_reportable.sum()),
            min_auc=float(finite.roc_auc.min()) if len(finite) else np.nan,max_brier=float(finite.brier.max()) if len(finite) else np.nan,
            max_abs_rdsi=float(finite.r_dsi.abs().max()) if len(finite) else np.nan,max_abs_s_minus1=float((finite.s_theta-1).abs().max()) if len(finite) else np.nan))
    csv('near_perfect_boundary_summary.csv',pd.DataFrame(bsummary))
    captures=[];monotonic=[]
    for key,g in cond.groupby(KEY,sort=False):
        g=g.sort_values('capture');assert len(g)==5
        positive=g[g.capture>0]
        finite=truth(g.numeric_finite).all()
        monotonic.append(dict(zip(KEY,key),scenario=g.scenario.iloc[0],all_capture_finite=finite,
            positive_capture_auc_spread=float(positive.roc_auc.max()-positive.roc_auc.min()),
            rdsi_drop_gt_002=bool(np.any(np.diff(g.r_dsi.to_numpy(float)) < -.02)) if finite else False,
            constant_capture_rdsi=g[g.capture==0].r_dsi.iloc[0],
            full_capture_rdsi=g[g.capture==1].r_dsi.iloc[0]))
    monotonic=pd.DataFrame(monotonic)
    csv('capture_response_checks.csv',monotonic)
    condition_groups={key:g for key,g in cond.groupby(['N','arm','fold_count','scenario','capture'])}
    for key,requested in data.groupby(['N','arm','fold_count','scenario']):
        n,arm,cv,scene=key
        assert len(requested)==100
        for c in run.CAPTURES:
            g=condition_groups.get((*key,c),cond.iloc[:0])
            finite=g[truth(g.numeric_finite)]
            row=dict(N=int(n),arm=arm,fold_count=int(cv),scenario=scene,capture=float(c),worlds=100,
                sampling_feasible=len(g),finite=len(finite),fully_reportable=int(truth(g.fully_reportable).sum()))
            for field in ['r_dsi','s_theta','roc_auc','brier']:
                vals=finite[field].to_numpy(float)
                row[field+'_median']=float(np.median(vals)) if len(vals) else np.nan
            captures.append(row)
    csv('capture_response_summary.csv',pd.DataFrame(captures))
    # Actual sample/fold information helps separate size, density, strata and split changes.
    structure=fold.groupby(['N','arm','fold_count','scenario']).agg(fold_records=('fold','size'),median_validation_n=('validation_n','median'),median_validation_positive=('validation_positive','median'),median_tree_H=('tree_H','median'),median_kmeans_H=('kmeans_H','median')).reset_index()
    csv('fold_structure_summary.csv',structure)
    graph=tables['graphs'].groupby(['N','arm','fold_count','scenario','k']).agg(median_neighbor_distance_m=('distance_median','median'),median_p95_neighbor_distance_m=('distance_p95','median')).reset_index()
    csv('graph_distance_summary.csv',graph)
    # Independent recomputation of Holm from six raw p values in every valid fold.
    holm_errors=[]
    for key,g in screen.groupby(KEY+['fold'],sort=False):
        if len(g)==6 and truth(g.screen_valid).all() and np.isfinite(g.permutation_p).all():
            p=g.permutation_p.to_numpy();order=np.argsort(p);a=np.empty(6);a[order]=np.minimum(1,np.maximum.accumulate(p[order]*(6-np.arange(6))))
            holm_errors.append(float(np.max(np.abs(a-g.holm_p.to_numpy()))))
    assert max(holm_errors,default=0)<1e-12
    protocol=json.loads((OUT/'replacement_protocol.json').read_text(encoding='utf-8'))
    source_unchanged=all(digest(Path(p))==h for p,h in protocol['protected_inputs'].items())
    assert source_unchanged
    selected=boundary[(boundary.N.isin([900,3600]))&boundary.numeric_finite]
    verification=dict(success=True,utc=now(),original_inputs_unchanged=True,
        independent_worlds=600,worlds_per_scenario=100,original_dataset_rows=9600,added_size_dataset_rows=4800,
        dataset_rows=len(data),capture_records=len(cond),fold_records=len(fold),response_test_records=len(screen),
        all_main_counts_reconciled=True,holm_valid_folds_rechecked=len(holm_errors),max_holm_difference=max(holm_errors,default=0),
        boundary_requested_dataset_controls=len(boundary),boundary_sample_feasible=int(boundary.sampling_feasible.sum()),boundary_finite=int(boundary.numeric_finite.sum()),
        boundary_main_sizes_finite=len(selected),boundary_main_sizes_max_abs_rdsi=float(selected.r_dsi.abs().max()),
        boundary_main_sizes_max_brier=float(selected.brier.max()),boundary_main_sizes_min_auc=float(selected.roc_auc.min()),
        max_positive_capture_auc_spread=float(monotonic.positive_capture_auc_spread.max()),
        numerical_complete_across_all_capture_datasets=int(monotonic.all_capture_finite.sum()),
        rdsi_drop_gt_002_among_complete=int(monotonic.rdsi_drop_gt_002.sum()),
        manuscript_modified=False,formulas_changed=False,main_alpha_changed=False)
    dump('verification.json',verification)
    print(json.dumps(verification,ensure_ascii=False,indent=2),flush=True)
    print(totals.to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','extend','boundary','summarize']);p.add_argument('--workers',type=int,default=8);p.add_argument('--smoke',action='store_true');p.add_argument('--ready-only',action='store_true');a=p.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.mode=='freeze':freeze();print('FROZEN',flush=True)
    elif a.mode=='extend':extend(a.workers,a.smoke)
    elif a.mode=='boundary':boundary(a.workers,a.smoke,a.ready_only)
    elif a.mode=='summarize':summarize()


if __name__=='__main__':main()
