from common import *
from scipy.stats import spearmanr

METRICS=['auc','brier','reference_theta','r_dsi','s_theta']
def complete(v):
 v=np.asarray(v,float);return float(v.mean()) if np.isfinite(v).all() else np.nan
def weighted(v,w):
 v=np.asarray(v,float);return float(np.average(v,weights=w)) if np.isfinite(v).all() else np.nan
def bootstrap(values):
 a=np.asarray(values,float)
 if not np.isfinite(a).all():return np.nan,np.nan
 rng=np.random.default_rng(20261013);samples=rng.integers(0,len(a),size=(20000,len(a)));b=a[samples].mean(axis=1)
 return tuple(np.quantile(b,[.025,.975],method='linear'))
def pareto(df):
 out=[]
 for i,row in df.iterrows():
  others=df.drop(index=i);dominates=(others.auc>=row.auc)&(others.brier<=row.brier)&(others.r_dsi>=row.r_dsi)&((others.auc>row.auc)|(others.brier<row.brier)|(others.r_dsi>row.r_dsi))
  out.append('Dominated' if dominates.any() else 'ND')
 return out

def summarize_sweeps():
 folds=read(ROOT/'tables/sweep_fold_results.csv');rows=[]
 for (ex,setting,seed),g in folds.groupby(['experiment','setting','seed'],sort=False):
  assert len(g)==5 and set(g.fold)==set(range(1,6))
  pred=pd.concat([read(p) for p in g.sort_values('fold').prediction_path]);assert pred.sample_id.is_unique and len(pred)==8806
  row={'experiment':ex,'setting':setting,'seed':seed,'mean_training_clusters':float(g.training_clusters.mean()),'min_training_clusters':int(g.training_clusters.min()),'max_training_clusters':int(g.training_clusters.max()),'n_finite_fold_theta':int(g.n_finite_theta.sum()),'all_folds_screen_pass':bool(g.fully_gated.all()),'all_folds_holm_pass':bool(g.holm_gated.all()),'reference_theta':weighted(g.theta_tree_raw,g.validation_n),'r_dsi':complete(g.r_dsi),'s_theta':complete(g.s_theta),'r_dsi_gated':complete(g.r_dsi_gated),'auc_equal_fold':complete(g.auc),'brier_equal_fold':complete(g.brier),**metrics(pred.label,pred.probability.to_numpy(float))}
  rows.append(row)
 seed=pd.DataFrame(rows);csv(seed,ROOT/'tables/sweep_seed_results.csv');summary=[];sensitivity=[]
 for ex in ['spatial','attribute']:
  df=seed.loc[seed.experiment.eq(ex)]
  configs=df.groupby('setting',sort=False)[METRICS+['mean_training_clusters','min_training_clusters','max_training_clusters','r_dsi_gated','auc_equal_fold','brier_equal_fold']].agg(complete).reset_index()
  configs['experiment']=ex;configs['pareto']=pareto(configs)
  equal=configs.copy();equal['auc']=equal.auc_equal_fold;equal['brier']=equal.brier_equal_fold;configs['pareto_equal_fold']=pareto(equal)
  summary.append(configs)
  reference=df.loc[df.setting.eq('all_events')].set_index('seed')
  for setting,g in df.groupby('setting',sort=False):
   g=g.set_index('seed').loc[SEEDS]
   for metric in METRICS:
    values=g[metric].to_numpy()-reference.loc[SEEDS,metric].to_numpy();lo,hi=bootstrap(values)
    sensitivity.append({'experiment':ex,'setting':setting,'metric':metric,'paired_seed_n':len(values),'difference':complete(values),'ci95_low':lo,'ci95_high':hi})
 summary=pd.concat(summary,ignore_index=True);csv(summary,ROOT/'tables/sweep_configuration_results.csv');csv(pd.DataFrame(sensitivity),ROOT/'tables/sweep_paired_intervals.csv')
 screen=folds[['experiment','seed','fold','fully_gated','holm_gated']].drop_duplicates();csv(screen,ROOT/'tables/sweep_response_screen_coverage.csv')
 for ex in ['spatial','attribute']:
  d=summary.loc[summary.experiment.eq(ex)];csv(d,ROOT/'tables'/f'Fig{8 if ex=="spatial" else 9}_plot_data.csv')
 return folds,seed,summary

def summarize_scales():
 folds=read(ROOT/'tables/Fig6_fold_results.csv');rows=[]
 for (fam,scale,seed),g in folds.groupby(['family','scale_m','seed']):
  pred=pd.concat([read(p) for p in g.prediction_path]);assert len(g)==5 and pred.sample_id.is_unique and len(pred)==8806
  rows.append({'family':fam,'scale_m':scale,'seed':seed,'mean_train_n':float(g.train_n.mean()),'minimum_train_validation_distance_m':float(g.minimum_train_validation_distance_m.min()),**metrics(pred.label,pred.probability.to_numpy(float))})
 seed=pd.DataFrame(rows);csv(seed,ROOT/'tables/Fig6_seed_results.csv');contrasts=[]
 for fam,reference_scale in [('radius',1000),('block',5000),('gap',0)]:
  frame=seed.loc[seed.family.eq(fam)];ref=frame.loc[frame.scale_m.eq(reference_scale)].set_index('seed')
  for scale,g in frame.groupby('scale_m'):
   g=g.set_index('seed').loc[SEEDS[:5]]
   for metric in ['auc','brier']:
    delta=g[metric]-ref.loc[SEEDS[:5],metric];contrasts.append({'family':fam,'scale_m':scale,'metric':metric,'seed_n':5,'absolute_mean':float(g[metric].mean()),'difference':float(delta.mean()),'difference_SD':float(delta.std(ddof=1)),'reference_scale_m':reference_scale})
 csv(pd.DataFrame(contrasts),ROOT/'tables/Fig6_plot_data.csv')

def summarize_importance():
 raw=read(ROOT/'tables/Fig10_permutation_raw.csv.gz')
 fold=raw.groupby(['seed','fold','feature','mode'],sort=False).agg(delta_auc=('delta_auc','mean'),delta_brier=('delta_brier','mean'),changed_row_fraction=('changed_rows','mean'),validation_n=('validation_n','first')).reset_index()
 fold['changed_row_fraction']/=fold.validation_n
 seed=fold.groupby(['seed','feature','mode'],sort=False)[['delta_auc','delta_brier','changed_row_fraction']].mean().reset_index();rows=[]
 for (feature,mode),g in seed.groupby(['feature','mode'],sort=False):
  for metric in ['delta_auc','delta_brier']:
   lo,hi=bootstrap(g[metric]);rows.append({'feature':feature,'mode':mode,'metric':metric,'mean':float(g[metric].mean()),'SD_seeds':float(g[metric].std(ddof=1)),'ci95_low':lo,'ci95_high':hi,'changed_row_fraction':float(g.changed_row_fraction.mean())})
 summary=pd.DataFrame(rows);csv(fold,ROOT/'tables/Fig10_permutation_fold_results.csv');csv(seed,ROOT/'tables/Fig10_permutation_seed_results.csv');csv(summary,ROOT/'tables/Fig10_plot_data.csv')
 ranks=summary.loc[summary.metric.eq('delta_auc')].pivot(index='feature',columns='mode',values='mean').reindex(FACTORS16)
 rho=float(spearmanr(ranks['global'],ranks.within_5km_block).statistic)
 dump(ROOT/'tables/Fig10_rank_diagnostics.json',{'global_vs_within_block_mean_rank_spearman':rho,'global_top5':ranks['global'].nlargest(5).to_dict(),'within_block_top5':ranks.within_5km_block.nlargest(5).to_dict()})

def main():
    folds, seed, summary = summarize_sweeps()
    summarize_importance()
    if (ROOT / 'tables/Fig6_fold_results.csv').exists():
        summarize_scales()
    print(summary[['experiment', 'setting', 'auc', 'brier', 'reference_theta', 'r_dsi', 's_theta', 'pareto', 'mean_training_clusters']].to_string(index=False))
    print('SCREEN', folds[['experiment', 'seed', 'fold', 'fully_gated', 'holm_gated']].drop_duplicates().groupby('experiment')[['fully_gated', 'holm_gated']].sum().to_dict())

if __name__ == '__main__':
    main()
