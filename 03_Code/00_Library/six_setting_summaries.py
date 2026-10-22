"""Keep every residual/stratification summary separate; seed-block bootstrap only."""
from pathlib import Path
import sys,os,json
sys.dont_write_bytecode=True
from project_config import *
BASE=ensure_output(DEPENDENCE)
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

def summarize_vector(v,counts):
    good=np.isfinite(v);x=v[good];weights=counts[:,good]
    if not len(x):return dict(n_finite=0,n_missing=len(v),mean=np.nan,median=np.nan,q1=np.nan,q3=np.nan,iqr=np.nan,min=np.nan,max=np.nan,lower_2p5=np.nan,upper_97p5=np.nan,mean_ci_lower=np.nan,mean_ci_upper=np.nan,median_ci_lower=np.nan,median_ci_upper=np.nan,below_zero=0,above_one=0)
    total=weights.sum(axis=1);valid=total>0
    means=np.full(len(total),np.nan);means[valid]=(weights@x)[valid]/total[valid]
    order=np.argsort(x);sx=x[order];cumulative=np.cumsum(weights[:,order],axis=1)
    lower=(total-1)//2;upper=total//2
    il=(cumulative>lower[:,None]).argmax(axis=1);iu=(cumulative>upper[:,None]).argmax(axis=1)
    medians=(sx[il]+sx[iu])/2;medians[~valid]=np.nan
    return dict(n_finite=len(x),n_missing=len(v)-len(x),mean=float(x.mean()),median=float(np.median(x)),q1=float(np.quantile(x,.25)),q3=float(np.quantile(x,.75)),iqr=float(np.quantile(x,.75)-np.quantile(x,.25)),min=float(x.min()),max=float(x.max()),lower_2p5=float(np.quantile(x,.025)),upper_97p5=float(np.quantile(x,.975)),mean_ci_lower=float(np.nanquantile(means,.025)),mean_ci_upper=float(np.nanquantile(means,.975)),median_ci_lower=float(np.nanquantile(medians,.025)),median_ci_upper=float(np.nanquantile(medians,.975)),below_zero=int((x<0).sum()),above_one=int((x>1).sum()))

def main():
    raw=pd.read_csv(BASE/'repetition_results.csv.gz',keep_default_na=False,na_values=[''])
    metrics=[f'eta_a_{r}' for r in ['raw','deviance','pearson']]+[f'{prefix}_{s}_{r}' for prefix in ['eta_h','theta'] for r in ['raw','deviance','pearson'] for s in ['tree','kmeans']]
    keys=['scenario','arm','strata_policy','capture_fraction'];rng=np.random.default_rng(20261010);dist=[];pair=[];error=0.
    with threadpool_limits(limits=1):
        for group,part in raw.groupby(keys,sort=False):
            reps=np.sort(part.repetition.unique());ix=rng.integers(0,len(reps),(10000,len(reps)))
            count=np.zeros((10000,len(reps)),np.int16);np.add.at(count,(np.repeat(np.arange(10000),len(reps)),ix.ravel()),1)
            for metric in metrics:
                mat=part.pivot(index='repetition',columns='loading',values=metric).reindex(index=reps,columns=[0.,.25,.5,.85]).to_numpy()
                for j,loading in enumerate([0.,.25,.5,.85]):
                    common=dict(zip(keys,group));common.update(loading=loading,metric=metric,aggregation_level='equal-fold mean per repetition; no pooling of folds',bootstrap_repetitions=10000,bootstrap_seed=20261010)
                    v=mat[:,j];delta=v-mat[:,0]
                    dist.append(dict(**common,**summarize_vector(v,count)))
                    pair.append(dict(**common,**summarize_vector(delta,count)))
                    # Independent direct-resampling check of weighted-median implementation.
                    if group[0]=='mixed' and metric=='theta_tree_raw' and loading==.85:
                        direct=np.nanmedian(v[ix[:100]],axis=1)
                        good=np.isfinite(v);w=count[:100,good];order=np.argsort(v[good]);sv=v[good][order]
                        cumulative=np.cumsum(w[:,order],axis=1);total=w.sum(axis=1)
                        calc=(sv[(cumulative>((total-1)//2)[:,None]).argmax(axis=1)]+sv[(cumulative>(total//2)[:,None]).argmax(axis=1)])/2
                        error=max(error,float(np.max(abs(calc-direct))))
            print('DONE six-setting summaries',group,flush=True)
    assert error<1e-10
    pd.DataFrame(dist).to_csv(BASE/'tables/six_setting_distribution_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pair).to_csv(BASE/'tables/six_setting_paired_delta_summary.csv',index=False,encoding='utf-8-sig')
    (BASE/'qa/six_setting_summary_verification.json').write_text(json.dumps(dict(success=True,distribution_rows=len(dist),paired_rows=len(pair),bootstrap_median_direct_check_max_error=error),indent=2),encoding='utf-8')
    print('COMPLETE',len(dist),len(pair),'rows; direct-bootstrap check error',error)

if __name__=='__main__':main()
