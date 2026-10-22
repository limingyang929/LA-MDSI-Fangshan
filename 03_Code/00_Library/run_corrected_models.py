"""Fresh paired fits with original69 folds, raw features and matched weight totals."""
from __future__ import annotations
from pathlib import Path
import os,sys,json,time,hashlib,subprocess,traceback,platform
from datetime import datetime,timezone
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
sys.dont_write_bytecode=True
from project_config import *
BASE=ensure_output(BENCHMARK)
import numpy as np,pandas as pd,scipy,sklearn,xgboost,joblib
from joblib import Parallel,delayed
from scipy.spatial import cKDTree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from threadpoolctl import threadpool_limits
import model_core as core
SCHEMES=['all_events','prototype_unweighted','prototype_weighted']
MODELS=['logistic','random_forest','rbf_svm','xgboost']
def json_values(obj):
    """Preserve native nonfinite parameter markers as text, not invented values."""
    if isinstance(obj,dict):return {k:json_values(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [json_values(v) for v in obj]
    if isinstance(obj,(float,np.floating)) and not np.isfinite(obj):return str(obj)
    if hasattr(obj,'item'):return obj.item()
    return obj

def dump(path,obj):
    path=Path(path);assert path.resolve().is_relative_to(BASE.resolve());path.parent.mkdir(parents=True,exist_ok=True)
    if path.name=='failure.json':
        with (path.parent/'failure_history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(json_values({'utc':datetime.now(timezone.utc).isoformat(),**obj}),ensure_ascii=False)+'\n')
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(json_values(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)

def mode(q):
    """Deterministic nominal mode, using first sorted category for ties."""
    return q.mode(dropna=True).iloc[0]

def representations(train):
    """Train-only prototype means/modes/counts within fixed original CL_ID."""
    pos=train.loc[train.label.eq(1)].copy();neg=train.loc[train.label.eq(0)].copy();cats=['Landcover','Soil','Lithology'];numeric=[f for f in FEATURES[20] if f not in cats]
    aggregations={f:'mean' for f in numeric+['x','y']};aggregations.update({f:mode for f in cats})
    prototypes=pos.groupby('inventory_cluster',sort=True).agg(aggregations).reset_index();sizes=pos.groupby('inventory_cluster').size();prototypes['cluster_size']=prototypes.inventory_cluster.map(sizes).to_numpy(int);prototypes['label']=1;prototypes['sample_id']=[f'P_{int(i):05d}' for i in prototypes.inventory_cluster]
    assert prototypes.cluster_size.sum()==len(pos)
    result={};N=len(train);P=len(pos);Q=len(neg);G=len(prototypes)
    for scheme in SCHEMES:
        rr=train.copy() if scheme=='all_events' else pd.concat([prototypes,neg],ignore_index=True)
        y=rr.label.eq(1).to_numpy();mass=np.ones(len(rr))
        if scheme=='prototype_weighted':mass[y]=rr.loc[rr.label.eq(1),'cluster_size'].to_numpy(float)
        weight=np.empty(len(rr))
        for c in (0,1):
            ix=rr.label.eq(c).to_numpy();weight[ix]=N/2*mass[ix]/mass[ix].sum()
        assert np.isclose(weight.sum(),N) and np.isclose(weight[y].sum(),N/2) and np.isclose(weight[~y].sum(),N/2)
        rr['sample_weight']=weight;result[scheme]=rr
    return result

def worker(buffer,seed,fold,out):
    """Fit all feature/representation/classifier units for one frozen validation fold."""
    directory=out/f'b{buffer}_s{seed}_f{fold}';directory.mkdir(parents=True,exist_ok=True)
    endpoint=directory/'completion.json'
    if endpoint.exists():
        status=json.loads(endpoint.read_text(encoding='utf-8'));assert status['success'] and status['fits']==24
        return {'buffer_m':buffer,'seed':seed,'fold':fold,'fits':24,'reused_this_run_checkpoint':True}
    started=time.perf_counter()
    try:
        data=pd.read_csv(INPUT,float_precision='round_trip').set_index('sample_id',drop=False)
        folds=pd.read_csv(FOLD_SOURCE,float_precision='round_trip');a=folds.loc[folds.experiment.eq('main')&folds.seed.eq(seed)].set_index('sample_id').loc[data.index]
        assert a.fold.nunique()==5 and a.validation_group.nunique()==69
        assert np.array_equal(data.spatial_group.to_numpy(str),a.validation_group.to_numpy(str))
        va=data.loc[a.fold.eq(fold)].copy();potential=data.loc[a.fold.ne(fold)].copy()
        d=cKDTree(va[['x','y']]).query(potential[['x','y']],workers=1)[0];keep=d>buffer if buffer else np.ones(len(potential),bool);tr=potential.loc[keep].copy()
        assert set(tr.index).isdisjoint(va.index) and set(tr.spatial_group).isdisjoint(va.spatial_group)
        assert tr.label.nunique()==va.label.nunique()==2
        if buffer:assert d[keep].min()>buffer
        distance=cKDTree(tr[['x','y']]).query(va[['x','y']],workers=1)[0]
        exclude=potential[['sample_id','label','x','y','spatial_group']].copy();exclude['nearest_validation_m']=d;exclude['retained_train']=keep;exclude.to_csv(directory/'training_exclusions.csv.gz',index=False,compression='gzip',float_format='%.17g')
        membership=tr.loc[tr.label.eq(1),['sample_id','inventory_cluster','x','y']].copy();membership.to_csv(directory/'training_membership.csv.gz',index=False,compression='gzip',float_format='%.17g')
        rep=representations(tr);training_hash=hashlib.sha256('|'.join(tr.sample_id).encode()).hexdigest();validation_hash=hashlib.sha256('|'.join(va.sample_id).encode()).hexdigest()
        for s,rr in rep.items():rr.to_csv(directory/f'representation_{s}.csv.gz',index=False,compression='gzip',float_format='%.17g')
        checkpoint=directory/'metrics.jsonl';records=[json.loads(s) for s in checkpoint.read_text(encoding='utf-8').splitlines()] if checkpoint.exists() else [];done={r['key'] for r in records}
        processors={};matrices={}
        with threadpool_limits(limits=1):
            for count,features in FEATURES.items():
                cats=[f for f in ('Landcover','Soil','Lithology') if f in features];num=[f for f in features if f not in cats]
                for scaled in (False,True):
                    pp=ColumnTransformer([('numeric',StandardScaler() if scaled else 'passthrough',num),('category',OneHotEncoder(handle_unknown='ignore',sparse_output=True),cats)])
                    pp.fit(tr[features]);processors[count,scaled]=pp
                    joblib.dump({'preprocessor':pp,'features':features,'fit_training_ids_sha256':training_hash},directory/f'preprocessor_{count}_{int(scaled)}.joblib',compress=3)
                    xv=pp.transform(va[features])
                    for scheme,rr in rep.items():matrices[count,scaled,scheme]=(pp.transform(rr[features]),xv)
            specs=[(count,scheme,model) for count in FEATURES for scheme in SCHEMES for model in MODELS];np.random.default_rng(seed*100+fold+buffer).shuffle(specs)
            for count,scheme,name in specs:
                key=f'b{buffer}_s{seed}_f{fold}_n{count}_{scheme}_{name}'
                if key in done:continue
                rr=rep[scheme];weights=rr.sample_weight.to_numpy(float);y=rr.label.to_numpy(int);scaled=name in ('logistic','rbf_svm');xt,xv=matrices[count,scaled,scheme]
                model=core.estimator(name,seed+fold);params=model.get_params();assert params.get('class_weight') is None and (name!='xgboost' or params.get('scale_pos_weight') in (None,1))
                start=time.perf_counter();model.fit(xt,y,sample_weight=weights);fit_seconds=time.perf_counter()-start;p=model.predict_proba(xv)[:,1]
                assert len(p)==len(va) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                prediction=va[['sample_id','label','x','y','spatial_group']].copy();prediction['probability']=p;prediction['raw_residual']=prediction.label-p;prediction['seed']=seed;prediction['fold']=fold;prediction['buffer_m']=buffer;prediction['factor_count']=count;prediction['scheme']=scheme;prediction['model']=name;prediction['nearest_training_m']=distance
                ppath=directory/f'{key}_predictions.csv.gz';prediction.to_csv(ppath,index=False,compression='gzip',float_format='%.17g')
                model_path=directory/f'{key}_model.joblib';joblib.dump({'model':model,'preprocessor':processors[count,scaled],'features':FEATURES[count],'model_seed':seed+fold,'training_ids_sha256':training_hash,'validation_ids_sha256':validation_hash,'weight_formula':'N/(2*class_base_mass) times base_mass','class_total':len(tr)/2,'resolved_parameters':params},model_path,compress=3)
                row={'key':key,'buffer_m':buffer,'seed':seed,'fold':fold,'factor_count':count,'scheme':scheme,'model':name,'model_seed':seed+fold,'train_N':len(tr),'train_P':int(tr.label.sum()),'train_Q':int(tr.label.eq(0).sum()),'represented_M':len(rr),'represented_P':int(rr.label.sum()),'validation_n':len(va),'validation_P':int(va.label.sum()),'validation_ids_sha256':validation_hash,'training_ids_sha256':training_hash,'removed_train_n':int((~keep).sum()),'class_positive_weight':float(weights[y==1].sum()),'class_negative_weight':float(weights[y==0].sum()),'total_weight':float(weights.sum()),'positive_weight_min':float(weights[y==1].min()),'positive_weight_max':float(weights[y==1].max()),'negative_weight_single':float(weights[y==0][0]),'nearest_training_min_m':float(distance.min()),'nearest_training_median_m':float(np.median(distance)),'fit_seconds':fit_seconds,'prediction_path':str(ppath),'prediction_sha256':sha(ppath),'model_path':str(model_path),'model_sha256':sha(model_path),**core.metrics(va.label.to_numpy(int),p)}
                with checkpoint.open('a',encoding='utf-8') as f:f.write(json.dumps(row,default=lambda v:v.item())+'\n')
                records.append(row);done.add(key)
        assert len(done)==24;result={'success':True,'buffer_m':buffer,'seed':seed,'fold':fold,'fits':len(done),'validation_n':len(va),'training_n':len(tr),'minimum_original_distance_m':float(distance.min()),'elapsed_seconds':time.perf_counter()-started,'utc':datetime.now(timezone.utc).isoformat()};dump(endpoint,result)
        print('DONE_FOLD',buffer,seed,fold,round(result['elapsed_seconds'],2),flush=True);return result
    except Exception as e:
        dump(directory/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'utc':datetime.now(timezone.utc).isoformat()});raise

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=6);args=p.parse_args();out=BASE/'corrected_models';out.mkdir(exist_ok=True)
    lock=out/'RUNNING.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd);begin=time.perf_counter()
    try:
        manifest={'data':str(INPUT),'data_sha256':sha(INPUT)}
        environment={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'xgboost':xgboost.__version__,'joblib':joblib.__version__,'cpu_count':os.cpu_count(),'worker_count':args.workers,'model_threads':1,'command':subprocess.list2cmdline([sys.executable,*sys.argv]),'input_sha256':manifest['data_sha256'],'fold_source':str(FOLD_SOURCE),'fold_source_sha256':sha(FOLD_SOURCE),'runner_sha256':sha(__file__),'protocol_sha256':sha(CONFIG/'01_Analysis_protocol.md'),'core_parameters_sha256':sha(core.__file__),'features':FEATURES,'factor_categories':['Landcover','Soil','Lithology'],'fixed_model_parameters':{m:core.estimator(m,42).get_params() for m in MODELS},'seeds':[42] if args.smoke else SEEDS,'folds':[1] if args.smoke else [1,2,3,4,5],'buffers_m':[0,800],'smoke':args.smoke,'started_utc':datetime.now(timezone.utc).isoformat()}
        ep=out/'environment.json'
        if ep.exists():
            old=json.loads(ep.read_text(encoding='utf-8'))
            if old['runner_sha256']!=environment['runner_sha256']:
                assert args.smoke and not list(out.glob('b*_s*_f*/metrics.jsonl')),'Cannot change code after any completed fit'
                dump(out/'setup_attempts'/('environment_'+old['runner_sha256'][:12]+'.json'),old);dump(ep,environment)
            assert all(json.loads(ep.read_text(encoding='utf-8'))[k]==environment[k] for k in ['input_sha256','fold_source_sha256','runner_sha256','protocol_sha256','core_parameters_sha256'])
        else:dump(ep,environment)
        units=[(b,s,f) for b in (0,800) for s in environment['seeds'] for f in environment['folds']]
        result=Parallel(n_jobs=args.workers,backend='loky')(delayed(worker)(b,s,f,out) for b,s,f in units)
        expected=len(units)*24;table=pd.concat([pd.read_json(path,lines=True) for path in sorted(out.glob('b*_s*_f*/metrics.jsonl'))],ignore_index=True);assert len(table)==expected and table.key.is_unique
        table.to_csv(out/'fold_metrics.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
        dump(out/'completion.json',{'success':True,'formal_fits':expected if not args.smoke else 0,'smoke_fits':expected if args.smoke else 0,'complete_fold_designs':len(units),'elapsed_seconds':time.perf_counter()-begin,'utc':datetime.now(timezone.utc).isoformat(),'source_files_unchanged':sha(FOLD_SOURCE)==environment['fold_source_sha256'] and sha(manifest['data'])==environment['input_sha256']})
        print('ALL_DONE',expected,round(time.perf_counter()-begin,2),flush=True)
    except Exception as e:
        dump(out/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
