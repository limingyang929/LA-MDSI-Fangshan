from common import *
import argparse,time,traceback
from joblib import Parallel,delayed
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

def group_geometry(data):
 st=read(DATA/'05_Inventory_cluster_centres.csv').set_index('CL_ID')
 xy=data[['x','y']].to_numpy(float).copy();pos=data.label.eq(1).to_numpy()
 xy[pos]=st.loc[data.loc[data.label.eq(1),'inventory_cluster'],['X_CENT','Y_CENT']].to_numpy()
 origin=xy.min(axis=0)
 g=np.floor((xy-origin)/5000).astype(int)
 assert np.array_equal(np.char.add(np.char.add(g[:,1].astype(str),'_'),g[:,0].astype(str)),data.spatial_group)
 return xy,origin

def prepare():
 data=load_data();gxy,origin=group_geometry(data);pool=read(DATA/'02_Background_candidates.csv.gz')
 directory=ROOT/'scales/inputs';directory.mkdir(parents=True,exist_ok=True)
 csv(data,directory/'fixed_main_data.csv.gz');units=[];allocation=[]
 sourcefolds=read(ENV['fold_source'])
 for seed in SEEDS[:5]:
  mainfold=sourcefolds.loc[sourcefolds.experiment.eq('main')&sourcefolds.seed.eq(seed)].set_index('sample_id').loc[data.index,'fold']
  rng=np.random.default_rng(seed);priority=rng.permutation(len(pool))
  for radius in [100,250,500,750,1000,1250,1500]:
   eligible=priority[pool.nearest_landslide_m.to_numpy()[priority]>=radius];assert len(eligible)>=4403
   neg=pool.iloc[eligible[:4403]].copy();neg['inventory_cluster']=np.nan
   dd=pd.concat([data.loc[data.label.eq(1)],neg],ignore_index=True).set_index('sample_id',drop=False)
   xy=dd[['x','y']].to_numpy(float).copy();p=dd.label.eq(1).to_numpy();xy[p]=gxy[data.label.eq(1)]
   grid=np.floor((xy-origin)/5000).astype(int);dd['spatial_group']=np.char.add(np.char.add(grid[:,1].astype(str),'_'),grid[:,0].astype(str))
   path=directory/f'radius_{radius}_s{seed}.csv.gz';csv(dd,path)
   folds=np.empty(len(dd),int)
   for fold,(_,vi) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(dd[FEATURES],dd.label,groups=dd.spatial_group),1):folds[vi]=fold
   csv(pd.DataFrame({'sample_id':dd.sample_id,'fold':folds}),directory/f'radius_{radius}_s{seed}_folds.csv')
   csv(neg[['sample_id','candidate_id','attempt_index','x','y','nearest_landslide_m']],directory/f'radius_{radius}_s{seed}_selected_negatives.csv.gz')
   allocation.append({'seed':seed,'radius_m':radius,'eligible_n':len(eligible),'selected_n':len(neg),'positive_n':4403,'minimum_selected_distance_m':float(neg.nearest_landslide_m.min())})
   for f in range(1,6):units.append(('radius',radius,seed,f))
  for width in [1000,2000,5000,10000]:
   if width==5000:ff=mainfold.to_numpy(int);groups=data.spatial_group.to_numpy()
   else:
    grid=np.floor((gxy-origin)/width).astype(int);groups=np.char.add(np.char.add(grid[:,1].astype(str),'_'),grid[:,0].astype(str));ff=np.empty(len(data),int)
    for fold,(_,vi) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(data[FEATURES],data.label,groups=groups),1):ff[vi]=fold
   csv(pd.DataFrame({'sample_id':data.sample_id,'fold':ff,'validation_group':groups}),directory/f'block_{width}_s{seed}_folds.csv')
   for f in range(1,6):units.append(('block',width,seed,f))
  for gap in [0,500,800,1000,2000]:
   csv(pd.DataFrame({'sample_id':data.sample_id,'fold':mainfold.to_numpy(),'validation_group':data.spatial_group}),directory/f'gap_{gap}_s{seed}_folds.csv')
   for f in range(1,6):units.append(('gap',gap,seed,f))
 csv(pd.DataFrame(allocation),ROOT/'tables/Fig6_pseudoabsence_allocation.csv')
 dump(ROOT/'scales/design.json',{'sample_n':8806,'positive_n':4403,'negative_n':4403,'seeds':SEEDS[:5],'origin_xy_m':origin.tolist(),'original_5km_group_geometry_verified':True,'fold_model_units':len(units),'fixed_main_data_sha256':sha(INPUT),'pool_sha256':sha(DATA/'02_Background_candidates.csv.gz')})
 return units

def worker(family,scale,seed,fold):
 out=ROOT/'scales/models'/f'{family}_{scale}_s{seed}_f{fold}';out.mkdir(parents=True,exist_ok=True)
 if (out/'completion.json').exists():return
 start=time.perf_counter()
 try:
  source=ROOT/'scales/inputs'
  data=read(source/(f'radius_{scale}_s{seed}.csv.gz' if family=='radius' else 'fixed_main_data.csv.gz')).set_index('sample_id',drop=False)
  ff=read(source/f'{family}_{scale}_s{seed}_folds.csv').set_index('sample_id').loc[data.index]
  va=data.loc[ff.fold.eq(fold)].copy();tr=data.loc[ff.fold.ne(fold)].copy();potential=len(tr)
  if family=='gap' and scale:
   distance=cKDTree(va[['x','y']]).query(tr[['x','y']],workers=1)[0];tr=tr.loc[distance>scale].copy()
  assert tr.label.nunique()==2 and set(tr.index).isdisjoint(va.index)
  group_col='validation_group' if 'validation_group' in ff else 'spatial_group'
  groups=ff.validation_group if group_col=='validation_group' else data.spatial_group
  assert set(groups.loc[tr.index]).isdisjoint(groups.loc[va.index])
  reused=False;model_path=None;bundle=None
  if family!='radius':
   buffer=int(scale) if family=='gap' else 0
   src=AUDIT/'corrected_models'/f'b{buffer}_s{seed}_f{fold}'
   mp=src/f'b{buffer}_s{seed}_f{fold}_n16_all_events_xgboost_model.joblib'
   if mp.exists():
    candidate=joblib.load(mp)
    if candidate['training_ids_sha256']==idhash(tr.sample_id) and candidate['validation_ids_sha256']==idhash(va.sample_id):bundle=candidate;model_path=mp;reused=True
  with threadpool_limits(limits=1):
   w=weights(tr,len(tr))
   if reused:pp=bundle['preprocessor'];model=bundle['model']
   else:
    pp=preprocessor(tr);model=estimator(seed+fold);model.fit(pp.transform(tr[FEATURES]),tr.label.to_numpy(int),sample_weight=w)
    model_path=out/'model.joblib';joblib.dump({'preprocessor':pp,'model':model,'features':FEATURES,'training_ids_sha256':idhash(tr.sample_id),'validation_ids_sha256':idhash(va.sample_id),'weight_sum':float(w.sum()),'class_total':len(tr)/2,'model_seed':seed+fold},model_path,compress=3)
   prob=model.predict_proba(pp.transform(va[FEATURES]))[:,1]
   native_error=None
   if reused:
    stored=read(model_path.with_name(model_path.name.replace('_model.joblib','_predictions.csv.gz')))
    assert np.array_equal(stored.sample_id,va.sample_id);native_error=float(np.max(abs(prob-stored.probability)));assert native_error==0
   assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
  nearest=cKDTree(tr[['x','y']]).query(va[['x','y']],workers=1)[0]
  if family=='gap' and scale:assert nearest.min()>scale
  pred=va[['sample_id','label','x','y','spatial_group']].copy();pred['probability']=prob;pred['nearest_training_m']=nearest;csv(pred,out/'predictions.csv.gz')
  csv(tr[['sample_id','label','x','y']],out/'training.csv.gz')
  result={'family':family,'scale_m':scale,'seed':seed,'fold':fold,'train_n':len(tr),'removed_train_n':potential-len(tr),'validation_n':len(va),'validation_positive_n':int(va.label.sum()),'validation_class_count':int(va.label.nunique()),'fold_auc_available':va.label.nunique()==2,'training_positive_n':int(tr.label.sum()),'total_weight':float(w.sum()),'class_positive_weight':float(w[tr.label.to_numpy()==1].sum()),'class_negative_weight':float(w[tr.label.to_numpy()==0].sum()),'minimum_train_validation_distance_m':float(nearest.min()),'model_reused':reused,'model_path':str(model_path),'model_sha256':sha(model_path),'native_prediction_error':native_error,'prediction_path':str(out/'predictions.csv.gz'),**metrics(va.label,prob)}
  csv(pd.DataFrame([result]),out/'fold_results.csv');dump(out/'completion.json',{'success':True,'model_reused':reused,'new_fits':int(not reused),'elapsed_seconds':time.perf_counter()-start})
  print('SCALE_DONE',family,scale,seed,fold,'reuse',reused,round(time.perf_counter()-start,2),flush=True)
 except Exception as e:dump(out/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--workers', type=int, default=5)
    a.add_argument('--smoke', action='store_true')
    x = a.parse_args()
    units = prepare()
    targets = [('gap', 0, 42, 1), ('block', 1000, 42, 1), ('radius', 1500, 42, 1)] if x.smoke else units
    Parallel(n_jobs=x.workers, backend='loky')((delayed(worker)(*u) for u in targets))
    if not x.smoke:
        files = list((ROOT / 'scales/models').glob('*/fold_results.csv'))
        assert len(files) == 400
        results = pd.concat([read(p) for p in files])
        assert len(results) == 400
        csv(results, ROOT / 'tables/Fig6_fold_results.csv')
        dump(ROOT / 'scales/completion.json', {'success': True, 'fold_model_units': 400, 'reused_model_units': int(results.model_reused.sum()), 'new_fits': int((~results.model_reused).sum())})

if __name__ == '__main__':
    main()
