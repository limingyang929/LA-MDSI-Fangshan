from project_config import *
import os,sys,json,hashlib,ast
import numpy as np,pandas as pd,joblib
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage,fcluster
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score,brier_score_loss
from xgboost import XGBClassifier
from types import SimpleNamespace


ROOT=ensure_output(EXTENSIONS)
AUDIT=ensure_output(BENCHMARK)
FACTORS16=COMMON16
EXTRA=EXTRA4
FACTORS20=FACTORS16+EXTRA
ENV={'features':{str(k):v for k,v in FEATURES.items()},'fold_source':str(FOLD_SOURCE)}
FEATURES=ENV['features']['16']
CAT=['Landcover','Soil'];NUM=[v for v in FEATURES if v not in CAT]
K_VALUES=[4,8,16,32];RESIDUALS=['raw','deviance','pearson'];EPS=1e-12;P_CLIP=1e-6
sc=SimpleNamespace(K_VALUES=K_VALUES)
THETA=[f'theta_{s}_{r}' for r in RESIDUALS for s in ['tree','kmeans']]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def read(p):return pd.read_csv(p,keep_default_na=False,na_values=['NA',''],float_precision='round_trip')
def csv(df,p):
 p=Path(p);assert p.resolve().is_relative_to(ROOT.resolve());p.parent.mkdir(parents=True,exist_ok=True)
 df.to_csv(p,index=False,float_format='%.17g',na_rep='NA',encoding='utf-8-sig',compression={'method':'gzip','mtime':0} if p.suffix=='.gz' else None)
def dump(p,data):
 p=Path(p);assert p.resolve().is_relative_to(ROOT.resolve());p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
def load_data():return read(INPUT).set_index('sample_id',drop=False)
def idhash(ids):return hashlib.sha256('|'.join(map(str,ids)).encode()).hexdigest()
def metrics(y,p):return {'auc':float(roc_auc_score(y,p)) if np.unique(y).size==2 else np.nan,'brier':float(np.mean((np.asarray(y)-p)**2))}
def estimator(seed):
 return XGBClassifier(n_estimators=500,learning_rate=.1,max_depth=5,min_child_weight=8,gamma=1,subsample=.7,colsample_bytree=.7,reg_alpha=.5,reg_lambda=8,objective='binary:logistic',eval_metric='logloss',tree_method='hist',random_state=int(seed),n_jobs=1)
def preprocessor(train):
 pp=ColumnTransformer([('numeric','passthrough',NUM),('category',OneHotEncoder(handle_unknown='ignore',sparse_output=True),CAT)])
 return pp.fit(train[FEATURES])
def weights(frame,N):
 w=np.empty(len(frame));y=frame.label.to_numpy(int);mass=frame.cluster_size.to_numpy(float) if 'cluster_size' in frame else np.ones(len(frame))
 for cl in (0,1):
  ix=y==cl;w[ix]=N/2*mass[ix]/mass[ix].sum()
 assert np.isclose(w.sum(),N) and all(np.isclose(w[y==c].sum(),N/2) for c in [0,1])
 return w
def representation(train,labels):
 pos=train.loc[train.label.eq(1)].copy();neg=train.loc[train.label.eq(0)].copy()
 assert len(labels)==len(pos)
 pos['new_cluster']=labels;cats=['Landcover','Soil','Lithology']
 numeric=[x for x in ENV['features']['20']+['x','y'] if x not in cats]
 agg={v:'mean' for v in numeric};agg.update({v:lambda x:x.mode().iloc[0] for v in cats})
 proto=pos.groupby('new_cluster',sort=True).agg(agg).reset_index();sizes=pos.groupby('new_cluster').size()
 proto['cluster_size']=proto.new_cluster.map(sizes);proto['label']=1;proto['sample_id']=[f'P_{int(k):05d}' for k in proto.new_cluster]
 neg['cluster_size']=1
 rep=pd.concat([proto,neg],ignore_index=True);assert proto.cluster_size.sum()==len(pos)
 return rep,weights(rep,len(train))

# Extract the reviewed pure functions without running source-module global code.
def import_functions(path,names):
 text=Path(path).read_text(encoding='utf-8');parsed=ast.parse(text)
 nodes=[n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),globals())
import_functions(CODE/'scientific_core.py',['fit_real_strata','identifiability','theta_six','moran','q_stat','eta','joint','residual'])
import_functions(CODE/'graph_core.py',['correct_graphs','screen_holm'])

def diagnostics(train,validation,seed,directory,screen_seed=None):
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
 tree,km,rules=fit_real_strata(train,validation,seed,directory/'strata_rules.joblib')
 graphs,ga=correct_graphs(validation[['x','y']].to_numpy(float));levels={'tree':tree,'kmeans':km}
 for k,g in graphs.items():sparse.save_npz(directory/f'graph_k{k}.npz',g)
 screen=screen_holm(identifiability(validation.label.to_numpy(int),graphs,levels,499,seed*100 if screen_seed is None else screen_seed))
 csv(pd.DataFrame(screen),directory/'response_screen.csv')
 strata=validation[['sample_id','label','x','y']].copy();strata['tree_stratum']=tree;strata['kmeans_stratum']=km;csv(strata,directory/'strata.csv.gz')
 dump(directory/'diagnostic_rules.json',{'training_ids_hash':idhash(train.sample_id),'validation_ids_hash':idhash(validation.sample_id),'rules':rules,'graphs':ga,'nominal_all_six':all(x['identified'] for x in screen),'holm_all_six':all(x['holm_identified'] for x in screen)})
 return graphs,levels,screen

def clustering_inputs(pos):
 """Twenty environmental attributes; aspect counts once, nominal codes require equality."""
 n=len(pos);N=n*(n-1)//2
 fails={q:np.zeros(N,dtype=np.uint8) for q in [.05,.10,.15,.20]};tolerances={q:{} for q in fails}
 for f in FACTORS20:
  v=pos[f].to_numpy(float);d=pdist(v[:,None])
  if f in ['Landcover','Soil','Lithology']:
   fail=d>0
   for q in fails:fails[q]+=fail;tolerances[q][f]='exact nominal match'
  else:
   if f=='Aspect':
    flat=v<0;different_flat=pdist(flat[:,None].astype(float))>0;d=np.minimum(d,360-d);d[different_flat]=np.inf
   positive=d[(d>0)&np.isfinite(d)]
   values=np.quantile(positive,list(fails),method='linear') if len(positive) else np.repeat(1e-8,len(fails))
   for q,t in zip(fails,values):
    t=float(max(t,1e-8));fails[q]+=(d>t);tolerances[q][f]=t
 return pdist(pos[['x','y']].to_numpy(float)),fails,tolerances
def cluster(spatial,fail,cutoff,pass_ratio):
 distance=spatial/float(cutoff);distance=distance.copy();distance[fail>len(FACTORS20)-int(np.ceil(pass_ratio*len(FACTORS20)))]=2
 return fcluster(linkage(distance,method='complete'),1,criterion='distance').astype(int)

def frozen_folds(experiment,seed):
    table=read(CLUSTER_FOLDS)
    frame=table.loc[table.experiment.eq(experiment)&table.seed.eq(seed)]
    assert frame.sample_id.is_unique and len(frame)==8806 and frame.fold.nunique()==5
    return frame.set_index('sample_id').loc[load_data().index,'fold']
