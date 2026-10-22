import numpy as np
import hashlib
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import scientific_core as sc
def correct_graphs(xy):
    xy=np.asarray(xy,float);n=len(xy);result={};distances={}
    distance,index=NearestNeighbors(n_neighbors=33).fit(xy).kneighbors(xy)
    stripped=np.empty((n,32),int);d=np.empty((n,32))
    for i in range(n):
        keep=index[i]!=i
        assert keep.sum()>=32
        stripped[i]=index[i,keep][:32];d[i]=distance[i,keep][:32]
    for k in sc.K_VALUES:
        ix=stripped[:,:k]
        assert not np.any(ix==np.arange(n)[:,None])
        directed=sparse.csr_matrix((np.ones(n*k),(np.repeat(np.arange(n),k),ix.ravel())),shape=(n,n))
        assert np.all(np.diff(directed.indptr)==k)
        graph=directed.maximum(directed.T).tocsr()
        assert np.all(graph.diagonal()==0) and (graph-graph.T).nnz==0
        result[k]=graph;distances[k]={'distance_median':float(np.median(d[:,:k])),'distance_p95':float(np.quantile(d[:,:k],.95,method='linear')),'edge_count':graph.nnz,'graph_sha256':hashlib.sha256(graph.indptr.tobytes()+graph.indices.tobytes()).hexdigest()}
    return result,distances

def screen_holm(rows):
    p=np.array([r['permutation_p'] for r in rows],float);valid=np.isfinite(p)
    adjusted=np.full(len(p),np.nan)
    if valid.all():
        order=np.argsort(p);adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    for i,row in enumerate(rows): row['holm_p']=adjusted[i];row['holm_identified']=bool(np.isfinite(adjusted[i]) and adjusted[i]<.05)
    return rows
