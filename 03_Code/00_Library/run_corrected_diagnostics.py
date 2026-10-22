from pathlib import Path
from datetime import datetime,timezone
import os,sys,time,json,traceback,subprocess
from project_config import *
import numpy as np,pandas as pd,joblib
from joblib import Parallel,delayed
from scipy import sparse
from threadpoolctl import threadpool_limits
from run_corrected_models import FEATURES,dump,sha,FOLD_SOURCE
import scientific_core as sc
from graph_core import correct_graphs,screen_holm
BASE=ensure_output(BENCHMARK)
SOURCE=BASE/'corrected_models'
OUT=BASE/'corrected_diagnostics'
def worker(directory):
    name=directory.name;out=OUT/name;out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    if (out/'completion.json').exists():return {'fold':name,'checkpoint':True}
    try:
        data=pd.read_csv(INPUT,float_precision='round_trip').set_index('sample_id',drop=False)
        ex=pd.read_csv(directory/'training_exclusions.csv.gz',float_precision='round_trip');tr=data.loc[ex.loc[ex.retained_train,'sample_id']]
        units=pd.read_json(directory/'metrics.jsonl',lines=True);assert len(units)==24 and units.key.is_unique
        pred=pd.read_csv(units.iloc[0].prediction_path,float_precision='round_trip');va=data.loc[pred.sample_id];assert np.array_equal(va.label,pred.label)
        sc.FEATURES=FEATURES[16].copy();sc.CAT=['Landcover','Soil'];sc.NUM=[f for f in sc.FEATURES if f not in sc.CAT]
        with threadpool_limits(limits=1):
            tree,km,rules=sc.fit_real_strata(tr,va,int(units.iloc[0].seed+units.iloc[0].fold),out/'strata_rules.joblib');levels={'tree':tree,'kmeans':km}
            strata=va[['sample_id','label','x','y']].copy();strata['tree_stratum']=tree;strata['kmeans_stratum']=km;strata.to_csv(out/'strata_labels.csv.gz',index=False,compression='gzip',float_format='%.17g')
            graphs,audit=correct_graphs(va[['x','y']].to_numpy(float))
            for k,g in graphs.items():sparse.save_npz(out/f'graph_k{k}.npz',g)
            screen_seed=int(units.iloc[0].seed*100+units.iloc[0].fold)
            screen=screen_holm(sc.identifiability(va.label.to_numpy(int),graphs,levels,499,screen_seed));pd.DataFrame(screen).to_csv(out/'response_identifiability.csv',index=False,encoding='utf-8-sig')
            gate=all(r['identified'] for r in screen);holm=all(r['holm_identified'] for r in screen);rows=[]
            for unit in units.itertuples(index=False):
                prediction=pd.read_csv(unit.prediction_path,float_precision='round_trip');assert np.array_equal(prediction.sample_id,va.sample_id) and np.array_equal(prediction.label,va.label)
                assert sha(unit.prediction_path)==unit.prediction_sha256
                y=va.label.to_numpy(int);p=prediction.probability.to_numpy(float);values=sc.theta_six(y,p,graphs,levels)
                row={'key':unit.key,'buffer_m':int(unit.buffer_m),'seed':int(unit.seed),'fold':int(unit.fold),'factor_count':int(unit.factor_count),'scheme':unit.scheme,'model':unit.model,'validation_n':len(va),'fully_gated':gate,'holm_gated':holm,'r_dsi_gated':values['r_dsi'] if gate else np.nan,'s_theta_gated':values['s_theta'] if gate else np.nan,'permutation_seed':screen_seed,'permutations':499,'prediction_sha256':unit.prediction_sha256,**rules,**values};rows.append(row)
        pd.DataFrame(rows).to_csv(out/'fold_diagnostics.csv',index=False,encoding='utf-8-sig',float_format='%.17g');dump(out/'completion.json',{'success':True,'fits_diagnosed':24,'permutations_executed':499,'fully_gated':gate,'holm_gated':holm,'graph_audit':audit,'training_features':sc.FEATURES,'elapsed_seconds':time.perf_counter()-start,'utc':datetime.now(timezone.utc).isoformat()})
        print('DIAGNOSTIC_FOLD',name,round(time.perf_counter()-start,2),flush=True);return {'fold':name,'elapsed_seconds':time.perf_counter()-start}
    except Exception as e:dump(out/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--until-complete',action='store_true');p.add_argument('--smoke',action='store_true');args=p.parse_args();OUT.mkdir(exist_ok=True);start=time.perf_counter()
    lock=OUT/'RUNNING.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        evidence={'command':subprocess.list2cmdline([sys.executable,*sys.argv]),'python':sys.version,'worker_count':args.workers,'runner_sha256':sha(__file__),'core_sha256':sha(sc.__file__),'graph_function_source':str(CODE/'graph_core.py'),'graph_function_source_sha256':sha(CODE/'graph_core.py'),'protocol_sha256':sha(CONFIG/'01_Analysis_protocol.md'),'model_environment_sha256':sha(SOURCE/'environment.json'),'strata':{'features':FEATURES[16],'n_train_only':True,'CART':{'criterion':'entropy','max_depth':8,'min_samples_split':100,'min_samples_leaf':50,'ccp_alpha':.001},'Kmeans':{'K':5,'n_init':5,'max_iter':100,'rtol':1e-6,'atol':1e-8,'empty_cluster':'retain prior center'}},'permutations':499,'permutation_object':'unrestricted validation labels; fixed coordinates, neighbor graph and strata; class count preserved','primary_gate':'each of four Moran and two q responses nominal p<0.05; Holm supplementary','epsilon':1e-12,'probability_clipping':[1e-6,1-1e-6],'theta_formula':'1-(1-eta_A)*(1-eta_H)','component_or_theta_clipping':False,'aggregation':'within-fold six-setting linear Q25 and 1-IQR; mean5 folds then mean10 seeds; nonfinite retained','time_utc':datetime.now(timezone.utc).isoformat()}
        ep=OUT/'environment.json'
        if ep.exists():
            old=json.loads(ep.read_text(encoding='utf-8'));assert all(old[k]==evidence[k] for k in ['runner_sha256','core_sha256','protocol_sha256','model_environment_sha256'])
        else:dump(ep,evidence)
        while True:
            ready=[p.parent for p in sorted(SOURCE.glob('b*_s*_f*/completion.json'))];pending=[p for p in ready if not (OUT/p.name/'completion.json').exists()]
            if pending:Parallel(n_jobs=args.workers,backend='loky')(delayed(worker)(p) for p in pending)
            if len(list(OUT.glob('b*_s*_f*/completion.json')))==(2 if SMOKE else 100):break
            if not args.until_complete:break
            if (SOURCE/'failure.json').exists() and not (SOURCE/'RUNNING.lock').exists():raise RuntimeError('Model run failed; diagnostics will not fabricate missing units')
            time.sleep(2)
        paths=sorted(OUT.glob('b*_s*_f*/fold_diagnostics.csv'));table=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in paths],ignore_index=True) if paths else pd.DataFrame();table.to_csv(OUT/'fold_diagnostics.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
        success=len(paths)==(2 if SMOKE else 100)
        dump(OUT/('completion.json' if success else 'pending.json'),{'success':success,'fold_designs':len(paths),'diagnosed_model_units':len(table),'expected_units':48 if SMOKE else 2400,'elapsed_seconds':time.perf_counter()-start,'utc':datetime.now(timezone.utc).isoformat()})
    except Exception as e:dump(OUT/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
