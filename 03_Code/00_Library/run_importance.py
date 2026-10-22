from common import *
import argparse,time,traceback
from joblib import Parallel,delayed
from threadpoolctl import threadpool_limits

def worker(seed,fold):
 out=ROOT/'importance'/f's{seed}_f{fold}';out.mkdir(parents=True,exist_ok=True)
 if (out/'completion.json').exists():return
 start=time.perf_counter();source=AUDIT/'corrected_models'/f'b0_s{seed}_f{fold}'
 model_path=source/f'b0_s{seed}_f{fold}_n16_prototype_weighted_xgboost_model.joblib'
 pred_path=source/f'b0_s{seed}_f{fold}_n16_prototype_weighted_xgboost_predictions.csv.gz'
 bundle=joblib.load(model_path);pred=read(pred_path);data=load_data();va=data.loc[pred.sample_id].copy()
 assert np.array_equal(pred.label,va.label)
 with threadpool_limits(limits=1):
  p=bundle['model'].predict_proba(bundle['preprocessor'].transform(va[FEATURES]))[:,1]
  assert np.max(abs(p-pred.probability))==0
  base=metrics(va.label,p);rows=[];unchanged=[]
  # Original within-block definition: fixed main 5-km grouping, including centre grouping.
  blocks=list(va.groupby('spatial_group',sort=False).indices.values())
  for fi,factor in enumerate(FACTORS16):
   columns=['Aspect_sin','Aspect_cos'] if factor=='Aspect' else [factor]
   for mode in ['global','within_5km_block']:
    frames=[];changed=[]
    for rep in range(10):
     rng=np.random.default_rng(seed*100000+fold*10000+fi*100+rep+(0 if mode=='global' else 1000000))
     perm=np.arange(len(va))
     if mode=='global':perm=rng.permutation(perm)
     else:
      for ix in blocks:perm[ix]=rng.permutation(ix)
     vf=va[FEATURES].copy();vf[columns]=va[columns].to_numpy()[perm]
     frames.append(vf);changed.append(int(np.any(vf[columns].to_numpy()!=va[columns].to_numpy(),axis=1).sum()))
    pp=bundle['model'].predict_proba(bundle['preprocessor'].transform(pd.concat(frames,ignore_index=True)))[:,1].reshape(10,len(va))
    for rep,pr in enumerate(pp):
     m=metrics(va.label,pr);rows.append({'seed':seed,'fold':fold,'mode':mode,'feature':factor,'repeat':rep+1,'validation_n':len(va),'baseline_auc':base['auc'],'baseline_brier':base['brier'],'permuted_auc':m['auc'],'permuted_brier':m['brier'],'delta_auc':base['auc']-m['auc'],'delta_brier':m['brier']-base['brier'],'changed_rows':changed[rep]})
  csv(pd.DataFrame(rows),out/'permutations.csv.gz')
 dump(out/'completion.json',{'success':True,'model_reused':True,'new_fits':0,'model_path':str(model_path),'model_sha256':sha(model_path),'prediction_path':str(pred_path),'prediction_sha256':sha(pred_path),'baseline_metrics':base,'native_prediction_error':0,'records':len(rows),'elapsed_seconds':time.perf_counter()-start})
 print('IMPORTANCE_DONE',seed,fold,round(time.perf_counter()-start,2),flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    targets = [(42, 1)] if a.smoke else [(s, f) for s in SEEDS for f in range(1, 6)]
    Parallel(n_jobs=a.workers, backend='loky')((delayed(worker)(s, f) for s, f in targets))
    if not a.smoke:
        pieces = list((ROOT / 'importance').glob('s*_f*/permutations.csv.gz'))
        assert len(pieces) == 50
        raw = pd.concat([read(p) for p in pieces])
        assert len(raw) == 16000
        csv(raw, ROOT / 'tables/Fig10_permutation_raw.csv.gz')
        dump(ROOT / 'importance/completion.json', {'success': True, 'models_reused': 50, 'new_fits': 0, 'permutation_records': len(raw)})

if __name__ == '__main__':
    main()
