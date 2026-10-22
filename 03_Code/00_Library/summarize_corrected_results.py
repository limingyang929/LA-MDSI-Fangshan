"""Paired OOF summaries and whole69-group uncertainty; never treat seeds as independent cases."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,time
from project_config import *
BASE=ensure_output(BENCHMARK)
from run_corrected_models import dump,sha
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score,brier_score_loss
SOURCE=BASE/'corrected_models';DIAG=BASE/'corrected_diagnostics'

def mean_complete(v):
    v=np.asarray(v,float);return float(v.mean()) if np.isfinite(v).all() else np.nan

def weighted_score(y,p):
    order=np.argsort(p,kind='mergesort');s=p[order];y=y[order];starts=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1]
    def evaluate(w):
        w=w[order];positive=np.add.reduceat(w*y,starts);negative=np.add.reduceat(w*(1-y),starts);P=positive.sum();Q=negative.sum()
        auc=float((positive*(np.cumsum(negative)-negative/2)).sum()/(P*Q)) if P and Q else np.nan
        return auc,float(np.average((y-s)**2,weights=w))
    return evaluate

def main():
    start=time.perf_counter();assert json.loads((SOURCE/'completion.json').read_text(encoding='utf-8'))['success'];assert json.loads((DIAG/'completion.json').read_text(encoding='utf-8'))['success']
    m=pd.read_csv(SOURCE/'fold_metrics.csv',float_precision='round_trip');d=pd.read_csv(DIAG/'fold_diagnostics.csv',float_precision='round_trip');assert len(m)==len(d)==2400
    merged=m.merge(d,on=['key','buffer_m','seed','fold','factor_count','scheme','model','validation_n'],validate='one_to_one',suffixes=('','_diag'));merged.to_csv(BASE/'corrected_all_fold_results.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    seed_rows=[];pair_checks=[];ensembles={};weight_rows=[]
    theta=[f'theta_{s}_{r}' for r in ['raw','deviance','pearson'] for s in ['tree','kmeans']];fields=['r_dsi','s_theta','r_dsi_gated','s_theta_gated','theta_tree_raw']+theta
    for (buffer,count,scheme,model,seed),q in merged.groupby(['buffer_m','factor_count','scheme','model','seed']):
        assert len(q)==5 and q.fold.nunique()==5
        p=pd.concat([pd.read_csv(path,float_precision='round_trip') for path in q.prediction_path],ignore_index=True).sort_values('sample_id');assert len(p)==8806 and p.sample_id.is_unique and p.label.sum()==4403
        row={'buffer_m':buffer,'factor_count':count,'scheme':scheme,'model':model,'seed':seed,'OOF_n':len(p),'OOF_P':int(p.label.sum()),'OOF_Q':int(p.label.eq(0).sum()),'roc_auc':float(roc_auc_score(p.label,p.probability)),'brier':float(brier_score_loss(p.label,p.probability)),'ece10':float(sum(abs(g.probability.mean()-g.label.mean())*len(g)/len(p) for _,g in p.assign(bin=np.minimum((p.probability*10).astype(int),9)).groupby('bin'))),'gate_folds':int(q.fully_gated.sum()),'holm_gate_folds':int(q.holm_gated.sum())}
        row.update({key:mean_complete(q[key]) for key in fields});seed_rows.append(row)
        key=(int(buffer),int(count),scheme,model);ensembles.setdefault(key,[]).append(p[['sample_id','label','x','y','spatial_group','probability']].set_index('sample_id'))
    seeds=pd.DataFrame(seed_rows);seeds.to_csv(BASE/'corrected_seed_results.csv',index=False,encoding='utf-8-sig',float_format='%.17g');summary=[]
    for key,q in seeds.groupby(['buffer_m','factor_count','scheme','model']):
        row=dict(zip(['buffer_m','factor_count','scheme','model'],key));assert len(q)==10;row['seeds']=10;row['OOF_n_each_seed']=8806;row['full_gate_folds_out_of_50']=int(q.gate_folds.sum());row['full_gate_seeds_out_of_10']=int(q.gate_folds.eq(5).sum())
        for field in ['roc_auc','brier','ece10']+fields:
            v=q[field].to_numpy(float);row[field+'_mean']=mean_complete(v);row[field+'_sd']=float(v.std(ddof=1)) if np.isfinite(v).all() else np.nan;row[field+'_finite_seeds']=int(np.isfinite(v).sum())
        summary.append(row)
    pd.DataFrame(summary).to_csv(BASE/'corrected_configuration_summary.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    mean_predictions={}
    for key,parts in ensembles.items():
        anchor=parts[0]
        assert all(np.array_equal(p.index,anchor.index) and np.array_equal(p.label,anchor.label) and np.array_equal(p.spatial_group,anchor.spatial_group) for p in parts)
        avg=anchor.copy();avg['probability']=np.stack([p.probability.to_numpy(float) for p in parts]).mean(axis=0);mean_predictions[key]=avg
        target=BASE/'mean_OOF_predictions'/f'b{key[0]}_n{key[1]}_{key[2]}_{key[3]}.csv.gz';target.parent.mkdir(exist_ok=True);avg.reset_index().to_csv(target,index=False,compression='gzip',float_format='%.17g')
    boot_rows=[];contrasts=[]
    for buffer in [0,800]:
        ref=next(p for key,p in mean_predictions.items() if key[0]==buffer);groups,gi=np.unique(ref.spatial_group.to_numpy(str),return_inverse=True);assert len(groups)==69
        rng=np.random.default_rng(20261010+buffer);counts=np.array([np.bincount(rng.integers(0,69,69),minlength=69) for _ in range(2000)])
        np.savez_compressed(BASE/f'bootstrap_whole69_b{buffer}.npz',group_labels=groups,draw_multiplicities=counts,seed=20261010+buffer)
        for scheme in ['all_events','prototype_unweighted','prototype_weighted']:
            for model in ['logistic','random_forest','rbf_svm','xgboost']:
                left=mean_predictions[buffer,16,scheme,model];right=mean_predictions[buffer,20,scheme,model];assert np.array_equal(left.index,right.index) and np.array_equal(left.label,right.label) and np.array_equal(left.spatial_group,right.spatial_group)
                y=left.label.to_numpy(float);f16=weighted_score(y,left.probability.to_numpy(float));f20=weighted_score(y,right.probability.to_numpy(float));delta=[]
                for repeat,c in enumerate(counts,1):
                    a,b=f16(c[gi]);x,z=f20(c[gi]);dr=[x-a,z-b];delta.append(dr);boot_rows.append({'buffer_m':buffer,'scheme':scheme,'model':model,'repeat':repeat,'resampling_seed':20261010+buffer,'delta_AUC':dr[0],'delta_Brier':dr[1]})
                delta=np.asarray(delta);original16=f16(np.ones(len(y)));original20=f20(np.ones(len(y)));qs=seeds.loc[seeds.buffer_m.eq(buffer)&seeds.scheme.eq(scheme)&seeds.model.eq(model)].pivot(index='seed',columns='factor_count',values=['roc_auc','brier','r_dsi','s_theta'])
                contrasts.append({'buffer_m':buffer,'scheme':scheme,'model':model,'comparison':'20 minus16','same_OOF_ids':True,'same_fixed_negatives':True,'same_folds':True,'same_training_members':True,'same_class_total_weight':True,'resampling_units':69,'bootstrap_repetitions':2000,'bootstrap_seed':20261010+buffer,'delta_AUC_mean_seed_metric':float((qs['roc_auc'][20]-qs['roc_auc'][16]).mean()),'delta_Brier_mean_seed_metric':float((qs['brier'][20]-qs['brier'][16]).mean()),'delta_R_DSI_fold_then_seed_mean':mean_complete(qs['r_dsi'][20]-qs['r_dsi'][16]),'delta_S_theta_fold_then_seed_mean':mean_complete(qs['s_theta'][20]-qs['s_theta'][16]),'delta_AUC_mean_OOF_probability':original20[0]-original16[0],'delta_Brier_mean_OOF_probability':original20[1]-original16[1],'AUC_mean_OOF_probability_ci_low':float(np.quantile(delta[:,0],.025)),'AUC_mean_OOF_probability_ci_high':float(np.quantile(delta[:,0],.975)),'Brier_mean_OOF_probability_ci_low':float(np.quantile(delta[:,1],.025)),'Brier_mean_OOF_probability_ci_high':float(np.quantile(delta[:,1],.975)),'interval_scope':'conditional paired69-group bootstrap of ten-seed mean OOF probabilities, not arithmetic seed-metric mean CI'})
    pd.DataFrame(boot_rows).to_csv(BASE/'paired_whole69_bootstrap_raw.csv.gz',index=False,compression='gzip',float_format='%.17g');pd.DataFrame(contrasts).to_csv(BASE/'corrected_16_20_paired_comparison.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    for (buffer,seed,fold,scheme,model),q in merged.groupby(['buffer_m','seed','fold','scheme','model']):
        assert len(q)==2 and q.factor_count.nunique()==2
        same=q.validation_ids_sha256.nunique()==q.training_ids_sha256.nunique()==1 and q.total_weight.nunique()==1
        assert same
        pair_checks.append({'buffer_m':buffer,'seed':seed,'fold':fold,'scheme':scheme,'model':model,'same_validation_training_hashes_and_weights':same,'validation_n':int(q.validation_n.iloc[0]),'training_n':int(q.train_N.iloc[0]),'class_weight_total':float(q.class_positive_weight.iloc[0])})
    pd.DataFrame(pair_checks).to_csv(BASE/'corrected_16_20_fold_pairing_audit.csv',index=False,encoding='utf-8-sig')
    for (buffer,seed,fold,scheme),q in m.groupby(['buffer_m','seed','fold','scheme']):
        anchor=q.iloc[0];weight_rows.append({k:anchor[k] for k in ['buffer_m','seed','fold','scheme','train_N','train_P','train_Q','represented_M','represented_P','class_positive_weight','class_negative_weight','total_weight','positive_weight_min','positive_weight_max','negative_weight_single','nearest_training_min_m']})
    pd.DataFrame(weight_rows).to_csv(BASE/'corrected_final_training_weights.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    dump(BASE/'summary_acceptance.json',{'success':True,'formal_new_fits':2400,'complete_seed_OOF_units':len(seeds),'configurations':len(summary),'paired_fold_checks':len(pair_checks),'paired_feature_contrasts':len(contrasts),'paired_bootstrap_rows':len(boot_rows),'bootstrap_repetitions_per_contrast':2000,'bootstrap_units':69,'summary_protocol_sha256':sha(CONFIG/'01_Analysis_protocol.md'),'script_sha256':sha(__file__),'not_all_components_bounded':[int(merged[c].lt(0).sum()) for c in theta],'nonfinite_RDSI_fold_records':int((~np.isfinite(merged.r_dsi)).sum()),'nonfinite_S_theta_fold_records':int((~np.isfinite(merged.s_theta)).sum()),'elapsed_seconds':time.perf_counter()-start,'utc':datetime.now(timezone.utc).isoformat()})
    print('SUMMARY_COMPLETE',len(seeds),len(summary),len(contrasts),len(boot_rows),flush=True)

if __name__=='__main__':main()
