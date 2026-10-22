from common import *
import argparse,time,traceback
from joblib import Parallel,delayed
from threadpoolctl import threadpool_limits

def settings(kind):
 if kind=='spatial':return [(f'cutoff_{c}',c,.1,.6) for c in [400,600,800,1000,1200]]
 return [('cutoff_800',800,.1,.6),('attr_q_0p05',800,.05,.6),('attr_q_0p15',800,.15,.6),('attr_q_0p2',800,.20,.6),('pass_0p5',800,.1,.5),('pass_0p7',800,.1,.7),('pass_0p8',800,.1,.8)]

def matched_main_control(train,va,seed,fold):
 src=AUDIT/'corrected_models'/f'b0_s{seed}_f{fold}'
 p=src/f'b0_s{seed}_f{fold}_n16_all_events_xgboost_model.joblib'
 if not p.exists():return None
 b=joblib.load(p)
 if b['training_ids_sha256']!=idhash(train.sample_id) or b['validation_ids_sha256']!=idhash(va.sample_id):return None
 assert b['features']==FEATURES and b['class_total']==len(train)/2
 return b,p

def worker(kind,seed,fold):
 out=ROOT/'sweeps'/kind/f's{seed}_f{fold}';out.mkdir(parents=True,exist_ok=True)
 if (out/'completion.json').exists():return
 started=time.perf_counter()
 try:
  data=load_data();ff=frozen_folds(kind,seed);va=data.loc[ff.eq(fold)].copy();tr=data.loc[ff.ne(fold)].copy();pos=tr.loc[tr.label.eq(1)].copy()
  assert set(tr.index).isdisjoint(va.index) and tr.label.nunique()==va.label.nunique()==2
  csv(va[['sample_id','label','x','y']],out/'validation.csv.gz');csv(tr[['sample_id','label','x','y']],out/'training.csv.gz')
  model_records=out/'metrics.jsonl';records=[json.loads(x) for x in model_records.read_text(encoding='utf-8').splitlines()] if model_records.exists() else []
  done={r['setting'] for r in records}
  with threadpool_limits(limits=1):
   pp=preprocessor(tr);xv=pp.transform(va[FEATURES]);graphs,levels,screen=diagnostics(tr,va,seed+fold,out/'diagnostics',seed*100+fold)
   gate=all(r['identified'] for r in screen);holm=all(r['holm_identified'] for r in screen)
   spatial,fails,tolerances=clustering_inputs(pos)
   dump(out/'attribute_tolerances.json',tolerances)
   for setting,cutoff,q,passing in [('all_events',0,0,0)]+settings(kind):
    if setting in done:continue
    reused=False;source_model=None
    if setting=='all_events':
     rr=tr.copy();w=weights(rr,len(tr));G=len(pos);labels=None;matched=matched_main_control(tr,va,seed,fold)
     if matched:
      bundle,source_model=matched;model=bundle['model'];prob=model.predict_proba(bundle['preprocessor'].transform(va[FEATURES]))[:,1];reused=True
    else:
     labels=cluster(spatial,fails[q],cutoff,passing);rr,w=representation(tr,labels);G=len(np.unique(labels))
     csv(pd.DataFrame({'sample_id':pos.sample_id.to_numpy(),'training_cluster':labels}),out/f'{setting}_membership.csv.gz')
    begin=time.perf_counter()
    if not reused:
     model=estimator(seed+fold);model.fit(pp.transform(rr[FEATURES]),rr.label.to_numpy(int),sample_weight=w);prob=model.predict_proba(xv)[:,1]
    elapsed=time.perf_counter()-begin
    assert len(prob)==len(va) and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
    model_path=out/f'{setting}_model.joblib'
    if not reused:joblib.dump({'model':model,'preprocessor':pp,'features':FEATURES,'training_ids_sha256':idhash(tr.sample_id),'validation_ids_sha256':idhash(va.sample_id),'training_clusters':G,'weight_sum':float(w.sum()),'class_weight_sums':[float(w[rr.label.to_numpy()==cl].sum()) for cl in [0,1]],'setting':setting,'model_seed':seed+fold},model_path,compress=3)
    else:model_path=source_model
    pred=va[['sample_id','label','x','y','spatial_group']].copy();pred['probability']=prob;csv(pred,out/f'{setting}_predictions.csv.gz')
    values=theta_six(va.label.to_numpy(int),prob,graphs,levels)
    row={'experiment':kind,'setting':setting,'cutoff_m':cutoff,'attr_q':q,'pass_ratio':passing,'seed':seed,'fold':fold,'validation_n':len(va),'train_n':len(tr),'train_positive_n':len(pos),'training_clusters':G,'represented_n':len(rr),'total_weight':float(w.sum()),'class_positive_weight':float(w[rr.label.to_numpy()==1].sum()),'class_negative_weight':float(w[rr.label.to_numpy()==0].sum()),'model_reused':reused,'model_path':str(model_path),'model_sha256':sha(model_path),'prediction_path':str(out/f'{setting}_predictions.csv.gz'),'fully_gated':gate,'holm_gated':holm,'fit_seconds':elapsed,**metrics(va.label,prob),**values}
    row['r_dsi_gated']=row['r_dsi'] if gate else np.nan
    with model_records.open('a',encoding='utf-8') as f:f.write(json.dumps(row,default=lambda v:v.item() if hasattr(v,'item') else str(v))+'\n')
    records.append(row);print('SWEEP_MODEL',kind,seed,fold,setting,round(elapsed,2),flush=True)
   assert len(records)==len(settings(kind))+1
   csv(pd.DataFrame(records),out/'fold_results.csv')
  dump(out/'completion.json',{'success':True,'records':len(records),'new_fits':sum(not r['model_reused'] for r in records),'validation_n':len(va),'response_all_six':gate,'holm_all_six':holm,'elapsed_seconds':time.perf_counter()-started,'protocol_sha256':sha(CONFIG/'01_Analysis_protocol.md')})
 except Exception as e:
  dump(out/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--workers', type=int, default=4)
    a.add_argument('--smoke', action='store_true')
    x = a.parse_args()
    targets = [('spatial', 42, 1)] if x.smoke else [(kind, s, f) for kind in ['spatial', 'attribute'] for s in SEEDS for f in range(1, 6)]
    Parallel(n_jobs=x.workers, backend='loky')((delayed(worker)(kind, s, f) for kind, s, f in targets))
    if not x.smoke:
        pieces = list((ROOT / 'sweeps').glob('*/*/fold_results.csv'))
        assert len(pieces) == 100
        df = pd.concat([read(p) for p in pieces])
        assert len(df) == 700
        csv(df, ROOT / 'tables/sweep_fold_results.csv')
        dump(ROOT / 'sweeps/completion.json', {'success': True, 'fold_designs': 100, 'model_units': 700, 'reused_model_units': int(df.model_reused.sum()), 'new_fits': int((~df.model_reused).sum())})

if __name__ == '__main__':
    main()
