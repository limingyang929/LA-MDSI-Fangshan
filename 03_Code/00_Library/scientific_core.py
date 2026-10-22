"""Task-local scientific calculations. No historical results are read here."""
from pathlib import Path
import os, sys
from project_config import *
BASE=ensure_output(LEGACY)
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'

import hashlib, importlib.util, json, logging, subprocess, time
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
import joblib

EPS=1e-12
P_CLIP=1e-6
SEEDS=[42,52,62,72,82,92,102,112,122,132]
MODELS=['logistic','random_forest','rbf_svm','xgboost']
SCHEMES=['all_events','cluster_centroid_unweighted','cluster_centroid_weighted']
BASE16=['Aspect','Curvature','Elevation','TWI','Slope','Landcover','Soil','Distance_to_river','Distance_to_road','FVC','NDVI_recent','NDVI_mean','NDVI_std','NDVI_trend','precipitation_mean','precipitation_max']
FEATURES=BASE16.copy()
CAT=['Landcover','Soil']
NUM=[x for x in FEATURES if x not in CAT]
RESIDUALS=['raw','deviance','pearson']
STRATA=['tree','kmeans']
K_VALUES=[4,8,16,32]
THETA=[f'theta_{s}_{r}' for r in RESIDUALS for s in STRATA]
STEPS=['A_DSI_signed','A1_magnitude_alignment','B_analytic_centering','C0_graph_convention','C_multiscale','D_residual_comparison','E_alternative_strata','F_six_components','G_lower_quartile','H_stability']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write_json(path,obj):
    path=Path(path)
    if not path.resolve().is_relative_to(BASE.resolve()):raise ValueError('Write outside new task directory')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

def logger(name):
    log=logging.getLogger(name)
    log.setLevel(logging.INFO)
    if not log.handlers:
        formatter=logging.Formatter('%(asctime)s %(message)s')
        for handler in (logging.FileHandler(BASE/f'run_{name}.log',encoding='utf-8'),logging.StreamHandler(sys.stdout)):
            handler.setFormatter(formatter);log.addHandler(handler)
    log.info('COMMAND %s',subprocess.list2cmdline([sys.executable,*sys.argv]))
    log.info('PYTHON %s',sys.version)
    return log

def block_ids(xy,size):
    a=np.floor(np.asarray(xy)/size).astype(np.int64)
    return np.array([f'{y}_{x}' for x,y in a])

def graph(xy,k,mode):
    """Original directed row-W; archived skip-first symmetric; corrected true symmetric."""
    xy=np.asarray(xy,float);n=len(xy)
    if n<=k+1:raise ValueError('Insufficient observations for frozen neighborhood')
    nn=NearestNeighbors(n_neighbors=k+1).fit(xy)
    index=nn.kneighbors(return_distance=False)[:,1:] if mode=='legacy' else nn.kneighbors(xy,return_distance=False)[:,1:]
    g=sparse.csr_matrix((np.ones(n*k),(np.repeat(np.arange(n),k),index.ravel())),shape=(n,n))
    if mode=='original':return g/k
    return g.maximum(g.T).tocsr()

def graphs(xy,mode='legacy'):
    return {k:graph(xy,k,mode) for k in K_VALUES}

def residual(y,p,kind):
    y=np.asarray(y,float);p=np.clip(p,P_CLIP,1-P_CLIP)
    if kind=='raw':return y-p
    if kind=='pearson':return (y-p)/np.sqrt(p*(1-p))
    if kind=='deviance':return np.sign(y-p)*np.sqrt(-2*np.where(y==1,np.log(p),np.log1p(-p)))
    raise ValueError(kind)

def moran(v,g):
    v=np.asarray(v,float);z=v-v.mean();den=z@z
    return float(len(v)/g.sum()*(z@g@z)/den) if den>0 and g.sum()>0 else np.nan

def q_stat(v,groups):
    v=np.asarray(v,float);_,code=np.unique(groups,return_inverse=True)
    h=code.max()+1;counts=np.bincount(code);sums=np.bincount(code,weights=v)
    den=np.sum((v-v.mean())**2)
    if den<=0 or h<2:return np.nan
    return float((np.sum(sums*sums/counts)-v.sum()**2/len(v))/den)

def eta(response,remainder,expectation=0,normalizer=1,signed=False):
    response=(response-expectation)/normalizer
    remainder=(remainder-expectation)/normalizer
    if abs(response)<=EPS:return np.nan
    return float(1-remainder/response if signed else 1-abs(remainder)/abs(response))

def joint(a,h):return float(1-(1-a)*(1-h))

def predictive(y,p):
    # Original 69-group source clips probabilities for its predictive metrics.
    y=np.asarray(y,int);p=np.clip(np.asarray(p,float),P_CLIP,1-P_CLIP)
    out=dict(roc_auc=float(roc_auc_score(y,p)),brier=float(brier_score_loss(y,p)),pr_auc=float(average_precision_score(y,p)),log_loss=float(log_loss(y,np.clip(p,P_CLIP,1-P_CLIP),labels=[0,1])),absolute_bias=float(abs(p.mean()-y.mean())))
    for bins in (5,10,20):
        z=np.minimum((p*bins).astype(int),bins-1)
        out[f'ece{bins}']=float(sum(abs(p[z==j].mean()-y[z==j].mean())*(z==j).mean() for j in range(bins) if np.any(z==j)))
    return out

def fit_strata_matrix(x_train,x_validation,y_train,seed):
    """Frozen CART and custom K=5 rules; validation labels are never accepted."""
    tree=DecisionTreeClassifier(criterion='entropy',max_depth=8,min_samples_split=100,min_samples_leaf=50,ccp_alpha=.001,random_state=seed)
    tree.fit(x_train,np.asarray(y_train,int))
    tree_labels=tree.apply(x_validation)
    scaler=StandardScaler().fit(x_train)
    tr=scaler.transform(x_train);va=scaler.transform(x_validation)
    rng=np.random.default_rng(seed);best_inertia=np.inf;best=None
    for start in range(5):
        centers=tr[rng.choice(len(tr),5,replace=False)].copy()
        for iteration in range(100):
            code=((tr[:,None,:]-centers[None,:,:])**2).sum(axis=2).argmin(axis=1)
            new=np.array([tr[code==j].mean(axis=0) if np.any(code==j) else centers[j] for j in range(5)])
            if np.allclose(new,centers,rtol=1e-6,atol=1e-8):centers=new;break
            centers=new
        inertia=float(np.min(((tr[:,None,:]-centers[None,:,:])**2).sum(axis=2),axis=1).sum())
        if inertia<best_inertia:best=centers.copy();best_inertia=inertia
    km=((va[:,None,:]-best[None,:,:])**2).sum(axis=2).argmin(axis=1)
    return tree_labels,km,dict(tree=tree,scaler=scaler,kmeans_centers=best,kmeans_inertia=best_inertia)

def fit_real_strata(train,validation,seed,path):
    """Use original train-standardized numerical and unscaled one-hot K-means convention."""
    pp=ColumnTransformer([('num','passthrough',NUM),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)])
    tr=pp.fit_transform(train[FEATURES]);va=pp.transform(validation[FEATURES])
    tree=DecisionTreeClassifier(criterion='entropy',max_depth=8,min_samples_split=100,min_samples_leaf=50,ccp_alpha=.001,random_state=seed).fit(tr,train.label)
    tree_labels=tree.apply(va)
    kp=ColumnTransformer([('num',StandardScaler(),NUM),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)])
    tr=kp.fit_transform(train[FEATURES]);va=kp.transform(validation[FEATURES])
    # K-means code is identical to the archived implementation, with no second scaling of one-hot columns.
    rng=np.random.default_rng(seed);best=None;best_inertia=np.inf
    for start in range(5):
        centers=tr[rng.choice(len(tr),5,replace=False)].copy()
        for iteration in range(100):
            code=((tr[:,None,:]-centers[None,:,:])**2).sum(axis=2).argmin(axis=1)
            new=np.array([tr[code==j].mean(axis=0) if np.any(code==j) else centers[j] for j in range(5)])
            if np.allclose(new,centers,rtol=1e-6,atol=1e-8):centers=new;break
            centers=new
        inertia=float(np.min(((tr[:,None,:]-centers[None,:,:])**2).sum(axis=2),axis=1).sum())
        if inertia<best_inertia:best=centers.copy();best_inertia=inertia
    km=((va[:,None,:]-best[None,:,:])**2).sum(axis=2).argmin(axis=1)
    path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(tree=tree,tree_preprocessor=pp,kmeans_preprocessor=kp,kmeans_centers=best,training_id_sha256=hashlib.sha256('|'.join(train.sample_id).encode()).hexdigest()),path)
    return tree_labels,km,dict(training_tree_leaves=int(tree.get_n_leaves()),validation_tree_levels=len(np.unique(tree_labels)),validation_kmeans_levels=len(np.unique(km)),kmeans_inertia=best_inertia)

def identifiability(y,g,levels,permutations,seed):
    """Separate 2-sided response permutation screen; analytic centering remains analytic."""
    y=np.asarray(y,float);rng=np.random.default_rng(seed)
    perm=np.stack([rng.permutation(y) for _ in range(permutations)],axis=1)
    z=perm-perm.mean(axis=0);den=(z*z).sum(axis=0)
    rows=[]
    for k,matrix in g.items():
        null=len(y)/matrix.sum()*(z*(matrix@z)).sum(axis=0)/den
        observed=moran(y,matrix);center=null.mean()
        valid=np.isfinite(observed) and np.isfinite(null).all() and np.isfinite(center)
        p=(1+np.sum(abs(null-center)>=abs(observed-center)))/(permutations+1) if valid else np.nan
        rows.append(dict(component='autocorrelation',setting=f'k{k}',observed=observed,null_mean=float(center),null_sd=float(null.std(ddof=1)),permutation_p=float(p),identified=bool(valid and p<.05),screen_valid=bool(valid),invalid_reason='' if valid else 'nonfinite_observed_or_null; not an identified response',permutations=permutations,permutation_seed=seed))
    for name,a in levels.items():
        uniq,code=np.unique(a,return_inverse=True);h=len(uniq)
        observed=q_stat(y,a)
        if h<2:
            rows.append(dict(component='heterogeneity',setting=name,observed=np.nan,null_mean=np.nan,null_sd=np.nan,permutation_p=np.nan,identified=False,permutations=permutations,permutation_seed=seed));continue
        counts=np.bincount(code)
        sums=np.zeros((h,permutations));np.add.at(sums,code,perm)
        null=(np.sum(sums*sums/counts[:,None],axis=0)-perm.sum(axis=0)**2/len(y))/den
        center=null.mean();valid=np.isfinite(observed) and np.isfinite(null).all() and np.isfinite(center)
        p=(1+np.sum(abs(null-center)>=abs(observed-center)))/(permutations+1) if valid else np.nan
        rows.append(dict(component='heterogeneity',setting=name,observed=observed,null_mean=float(center),null_sd=float(null.std(ddof=1)),permutation_p=float(p),identified=bool(valid and p<.05),screen_valid=bool(valid),invalid_reason='' if valid else 'nonfinite_observed_or_null; not an identified response',permutations=permutations,permutation_seed=seed))
    return rows

def theta_six(y,p,g,levels):
    n=len(y);ei=-1/(n-1);out={};iy={k:moran(y,w) for k,w in g.items()}
    for r in RESIDUALS:
        v=residual(y,p,r);ea=[]
        for k,w in g.items():
            ir=moran(v,w);a=eta(iy[k],ir,ei);ea.append(a)
            out.update({f'i_y_k{k}':iy[k],f'i_residual_k{k}_{r}':ir,f'eta_a_k{k}_{r}':a})
        a=float(np.mean(ea));out[f'eta_a_{r}']=a
        for s,groups in levels.items():
            qy=q_stat(y,groups);qr=q_stat(v,groups);eq=(len(np.unique(groups))-1)/(n-1)
            h=eta(qy,qr,eq,1-eq)
            out.update({f'q_y_{s}':qy,f'q_residual_{s}_{r}':qr,f'eta_h_{s}_{r}':h,f'theta_{s}_{r}':joint(a,h)})
    six=np.array([out[c] for c in THETA])
    out['n_finite_theta']=int(np.isfinite(six).sum())
    out['r_dsi']=float(np.quantile(six,.25,method='linear')) if np.isfinite(six).all() else np.nan
    out['s_theta']=float(1-(np.quantile(six,.75,method='linear')-np.quantile(six,.25,method='linear'))) if np.isfinite(six).all() else np.nan
    return out

def ablation(y,p,original_graph,legacy_graphs,levels):
    """Diagnostic ablation on one unchanged validation probability vector."""
    n=len(y);v=residual(y,p,'raw');iy=moran(y,original_graph);ir=moran(v,original_graph)
    qy=q_stat(y,levels['tree']);qr=q_stat(v,levels['tree']);eq=(len(np.unique(levels['tree']))-1)/(n-1)
    signed_a=eta(iy,ir,signed=True);signed_h=eta(qy,qr,signed=True)
    mag_a=eta(iy,ir);mag_h=eta(qy,qr)
    cent_a=eta(iy,ir,-1/(n-1));cent_h=eta(qy,qr,eq,1-eq)
    six=theta_six(y,p,legacy_graphs,levels)
    scores=[joint(signed_a,signed_h),joint(mag_a,mag_h),joint(cent_a,cent_h)]
    ca8=eta(moran(y,legacy_graphs[8]),moran(v,legacy_graphs[8]),-1/(n-1))
    scores += [joint(ca8,cent_h)]+[six['theta_tree_raw']]*4+[six['r_dsi']]*2
    rows=[]
    common=predictive(y,p)
    for ix,step in enumerate(STEPS):
        row={**common, 'step':step,'step_index':ix,'diagnostic_score':scores[ix], 'theta_reference':scores[ix] if ix<4 else six['theta_tree_raw'],'r_dsi':six['r_dsi'] if ix>=8 else np.nan,'s_theta':six['s_theta'] if ix>=9 else np.nan,'reported_combination_count':1 if ix<5 else 3 if ix==5 else 6,'undefined_summary_reason':'not_defined_before_this_step' if ix<8 else ('nonfinite_component' if six['n_finite_theta']<6 else '')}
        for c in THETA:
            row[c]=six[c] if ix>=6 or (ix==5 and '_tree_' in c) else np.nan
        row.update(dsi_eta_a_signed=signed_a,dsi_eta_h_signed=signed_h,dsi_theta1=min(signed_a,signed_h),dsi_theta2=max(signed_a,signed_h),dsi_theta3=joint(signed_a,signed_h))
        rows.append(row)
    return rows,six
