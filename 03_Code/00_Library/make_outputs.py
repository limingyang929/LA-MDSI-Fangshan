"""Publication plots, compact tables, and auditable results text from actual outputs."""
from pathlib import Path
import sys,os,json
sys.dont_write_bytecode=True
from project_config import *
BASE=ensure_output(DEPENDENCE)

os.environ['MPLCONFIGDIR']=str(BASE/'qa/matplotlib_cache')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ACTIVE=['weak_mixed','mixed','mixed_rare'];LAM=[0.,.25,.5,.85];CAP=[.25,.5,.75,1.]
LABELS={'weak_mixed':'Weak mixed (a = h = 0.3; prevalence = 0.20)',
        'mixed':'Mixed (a = h = 1.2; prevalence = 0.20)',
        'mixed_rare':'Rare mixed (a = h = 1.2; prevalence = 0.05)'}
COLORS=['#0072B2','#009E73','#E69F00','#CC79A7']

def sel(df,scene,arm,policy,capture,metric):
    return df.loc[df.scenario.eq(scene)&df.arm.eq(arm)&df.strata_policy.eq(policy)&df.capture_fraction.eq(capture)&df.metric.eq(metric)].sort_values('loading')
def fmt(v,d=3): return 'N/A' if not np.isfinite(v) else f'{v:.{d}f}'
def ci(m,lo,hi): return f'{fmt(m)} [{fmt(lo)}, {fmt(hi)}]'

def main():
    s=pd.read_csv(BASE/'tables/distribution_summary.csv',keep_default_na=False,na_values=[''])
    p=pd.read_csv(BASE/'tables/paired_delta_summary.csv',keep_default_na=False,na_values=[''])
    d=pd.read_csv(BASE/'generator_diagnostics.csv',keep_default_na=False,na_values=[''])
    r=pd.read_csv(BASE/'tables/capture_configuration_ranking.csv',keep_default_na=False,na_values=[''])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42})
    for family,metrics in [('LA_frozen',['eta_a','eta_h','theta']),('historical_fixed_bands',['hist_eta_a','hist_eta_h','hist_theta'])]:
        fig,axs=plt.subplots(3,3,figsize=(12.8,9.4),sharex=True)
        for i,scene in enumerate(ACTIVE):
            for j,m in enumerate(metrics):
                ax=axs[i,j]
                for c,color in zip(CAP,COLORS):
                    z=sel(s,scene,'fixed_amplitude','frozen',c,m)
                    x=z.loading.to_numpy();y=z['median'].to_numpy()
                    ax.plot(x,y,'o-',color=color,lw=1.6,ms=4)
                    ax.fill_between(x,z.median_ci_lower.to_numpy(),z.median_ci_upper.to_numpy(),color=color,alpha=.14)
                ax.axhline(0,color='.5',lw=.7,ls=':')
                ax.set_xticks(LAM);ax.grid(alpha=.16)
                ax.set_ylabel([r'$\eta_A$',r'$\eta_H$',r'$\theta$'][j])
                ax.set_title(['Autocorrelation component','Heterogeneity component','Product-complement aggregate'][j] if i==0 else '',fontsize=11)
                if j==0:ax.text(-.42,.5,LABELS[scene].replace('; ','\n').replace(' (','\n('),rotation=90,va='center',ha='center',transform=ax.transAxes,fontsize=9)
                if i==2:ax.set_xlabel(r'Generator template loading $\lambda$')
        handles=[Line2D([0],[0],color=color,marker='o',label=f'Capture c = {c:g}') for c,color in zip(CAP,COLORS)]
        fig.legend(handles=handles,loc='upper center',ncol=4,bbox_to_anchor=(.54,1.0),frameon=False)
        fig.subplots_adjust(left=.13,right=.98,bottom=.11,top=.92,wspace=.3,hspace=.28)
        text='LA-MDSI: raw residuals, multiscale Moran, frozen train-only CART, equal-fold means.' if family=='LA_frozen' else 'Historical diagnostic: full-grid k = 8, raw residuals, fixed five-band strata.'
        fig.text(.13,.035,text+'\nPoints: repetition medians. Bands: pointwise 95% bootstrap CIs of medians; finite scores only; 100 paired seed blocks per scenario.',fontsize=8.5)
        for ext in ['png','svg','pdf']:fig.savefig(BASE/'figures'/f'Fig_S_structure_dependence_{family}.{ext}',dpi=300)
        plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(11.3,8.9),sharex=True)
    for i,scene in enumerate(ACTIVE):
        for j,arm in enumerate(['fixed_amplitude','variance_matched']):
            ax=axs[i,j]
            for c,color in zip(CAP,COLORS):
                z=sel(p,scene,arm,'frozen',c,'delta_theta')
                ax.plot(z.loading,z['median'],'o-',color=color,lw=1.5,ms=4)
                ax.fill_between(z.loading.to_numpy(),z.median_ci_lower.to_numpy(),z.median_ci_upper.to_numpy(),color=color,alpha=.14)
            ax.axhline(0,color='.3',ls='--',lw=.8);ax.grid(alpha=.16);ax.set_xticks(LAM)
            ax.set_ylabel(r'Paired $\Delta\theta$')
            if i==0:ax.set_title(['Original amplitudes','Matched total logit-signal variance'][j],fontsize=11)
            if j==0:ax.text(-.28,.5,LABELS[scene].replace('; ','\n').replace(' (','\n('),rotation=90,ha='center',va='center',transform=ax.transAxes,fontsize=9)
            if i==2:ax.set_xlabel(r'Generator template loading $\lambda$')
    fig.legend(handles=handles,loc='upper center',ncol=4,bbox_to_anchor=(.55,1.),frameon=False)
    fig.subplots_adjust(left=.13,right=.98,bottom=.11,top=.92,wspace=.25,hspace=.28)
    fig.text(.13,.032,'LA-MDSI raw/CART, frozen strata. Points: medians of within-seed differences from loading 0.\nBands: pointwise 95% bootstrap CIs of paired medians. No component or score clipping.',fontsize=9)
    for ext in ['png','svg','pdf']:fig.savefig(BASE/'figures'/f'Fig_S_paired_theta_change.{ext}',dpi=300)
    plt.close(fig)
    rows=[]
    for scene in ACTIVE:
        for arm in ['fixed_amplitude','variance_matched']:
            for policy in ['frozen','adaptive']:
                for c in [0.,.25,.5,.75,1.]:
                    for lam in LAM:
                        z=s.loc[s.scenario.eq(scene)&s.arm.eq(arm)&s.strata_policy.eq(policy)&s.capture_fraction.eq(c)&s.loading.eq(lam)].set_index('metric')
                        dz=p.loc[p.scenario.eq(scene)&p.arm.eq(arm)&p.strata_policy.eq(policy)&p.capture_fraction.eq(c)&p.loading.eq(lam)].set_index('metric')
                        dd=d.loc[d.scenario.eq(scene)&d.arm.eq(arm)&d.loading.eq(lam)]
                        t=z.loc['theta'];da=z.loc['eta_a'];dh=z.loc['eta_h'];dt=dz.loc['delta_theta']
                        rows.append(dict(scenario=scene,arm=arm,strata_policy=policy,capture_fraction=c,loading=lam,
                           driver_r_median=dd.driver_correlation.median(),driver_r_q1=dd.driver_correlation.quantile(.25),driver_r_q3=dd.driver_correlation.quantile(.75),
                           n_eta_a=int(da.n_finite),n_eta_h=int(dh.n_finite),n_theta=int(t.n_finite),paired_n=int(dt.n_finite),
                           eta_a_median=da['median'],eta_h_median=dh['median'],theta_median=t['median'],theta_q1=t.q1,theta_q3=t.q3,
                           theta_min=t['min'],theta_max=t['max'],theta_median_ci_lower=t.median_ci_lower,theta_median_ci_upper=t.median_ci_upper,
                           delta_theta_mean=dt['mean'],delta_theta_mean_ci_lower=dt.mean_ci_lower,delta_theta_mean_ci_upper=dt.mean_ci_upper,
                           delta_theta_median=dt['median'],delta_theta_median_ci_lower=dt.median_ci_lower,delta_theta_median_ci_upper=dt.median_ci_upper))
    table=pd.DataFrame(rows);table.to_csv(BASE/'tables/main_results_all_captures.csv',index=False,encoding='utf-8-sig')
    compact=table.query("arm == 'fixed_amplitude' and strata_policy == 'frozen' and capture_fraction == 0.5")
    compact.to_csv(BASE/'tables/compact_primary_c05.csv',index=False,encoding='utf-8-sig')
    records=[]
    for row in compact.itertuples():
        records.append({'Scenario':row.scenario,'Loading':fmt(row.loading,2),'Realized r':fmt(row.driver_r_median),
            'n A/H/theta':f'{row.n_eta_a}/{row.n_eta_h}/{row.n_theta}',
            'Median eta A':fmt(row.eta_a_median),'Median eta H':fmt(row.eta_h_median),
            'Median theta [Q1,Q3]':ci(row.theta_median,row.theta_q1,row.theta_q3),
            'Paired n':str(row.paired_n),'Mean delta theta [95% MC CI]':ci(row.delta_theta_mean,row.delta_theta_mean_ci_lower,row.delta_theta_mean_ci_upper),
            'Median delta theta [95% MC CI]':ci(row.delta_theta_median,row.delta_theta_median_ci_lower,row.delta_theta_median_ci_upper)})
    (BASE/'tables/compact_table_formatted.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    report=[]
    report.append('Structure-dependence simulation: actual numerical findings\n')
    for scene in ACTIVE:
        f=table.query("scenario == @scene and strata_policy == 'frozen' and capture_fraction == .5 and loading == .85")
        for row in f.itertuples():
            report.append(f'{scene}, {row.arm}, c=0.5, loading=0.85: realized driver r median {row.driver_r_median:.6f}; paired n={row.paired_n}; mean delta theta {ci(row.delta_theta_mean,row.delta_theta_mean_ci_lower,row.delta_theta_mean_ci_upper)}; median delta theta {ci(row.delta_theta_median,row.delta_theta_median_ci_lower,row.delta_theta_median_ci_upper)}.')
    for scene in ACTIVE:
        for metric in ['theta','r_dsi','hist_theta']:
            q=r.loc[r.scenario.eq(scene)&r.arm.eq('fixed_amplitude')&r.strata_policy.eq('frozen')&r.metric.eq(metric)&r.loading.eq(.85)].iloc[0]
            report.append(f'{scene}, {metric}, five capture settings: n complete={q.complete_paired_repetitions}; rho={fmt(q.spearman_rho,6)}, tau_b={fmt(q.kendall_tau_b,6)}, reversals={q.reversed_pairs}/10, tie changes={q.tie_status_changes}.')
    (BASE/'actual_findings.txt').write_text('\n'.join(report),encoding='utf-8')
    print('\n'.join(report))

if __name__=='__main__':main()
