from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.7,'xtick.labelsize':7,'ytick.labelsize':7})
NAVY='#204080';BLUE='#5484C2';LIGHT='#94B8E0';ORANGE='#F2B279';REDORANGE='#E4724C';RED='#C82030';GRAY='#AAB5C8';GREEN='#248F83'
LABELS={'Aspect':'Aspect','Curvature':'Curvature','Elevation':'Elevation','TWI':'TWI','Slope':'Slope','Landcover':'Land cover','Soil':'Soil','Distance_to_river':'River distance','Distance_to_road':'Road distance','FVC':'FVC','NDVI_recent':'Recent NDVI','NDVI_mean':'Mean NDVI','NDVI_std':'NDVI SD','NDVI_trend':'NDVI trend','precipitation_mean':'Mean precipitation','precipitation_max':'Maximum precipitation'}
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
def save(fig,name,jpeg=False):
 for extension in ['png','pdf','svg','tiff']:
  fig.savefig(OUT/f'{name}.{extension}',dpi=600,facecolor='white')
 if jpeg:fig.savefig(OUT/f'{name}.jpeg',dpi=600,facecolor='white',pil_kwargs={'quality':95})
 plt.close(fig)
def err(ax,x,mid,lo,hi,color,**kwargs):
 ax.vlines(x,lo,hi,color=color,lw=1,alpha=.8);ax.plot([x-.03,x+.03],[lo,lo],color=color,lw=1);ax.plot([x-.03,x+.03],[hi,hi],color=color,lw=1);ax.plot(x,mid,'o',ms=4,color=color,**kwargs)

def fig8():
 data=read(ROOT/'tables/Fig8_plot_data.csv').set_index('setting');interval=read(ROOT/'tables/sweep_paired_intervals.csv')
 fig=plt.figure(figsize=(7.1,4.0));grid=fig.add_gridspec(3,2,width_ratios=[1.02,1],wspace=.43,hspace=.48,left=.10,right=.98,top=.91,bottom=.22)
 ax=fig.add_subplot(grid[:,0]);settings=[f'cutoff_{v}' for v in [400,600,800,1000,1200]];colors=[NAVY,BLUE,LIGHT,ORANGE,REDORANGE]
 coords=data.loc[settings,['auc','r_dsi']];ax.plot(coords.auc,coords.r_dsi,color=GRAY,lw=.8,zorder=1)
 offsets={400:(-24,-30),600:(-52,4),800:(-42,-27),1000:(8,-20),1200:(-15,11)}
 for key,color in zip(settings,colors):
  r=data.loc[key];ax.scatter(r.auc,r.r_dsi,s=r.mean_training_clusters/30,c=color,edgecolors=RED if r.pareto=='ND' else color,lw=1.1,zorder=3)
  threshold=int(key.split('_')[1]);dx,dy=offsets[threshold]
  ax.annotate(f'{threshold:,} m\nmean G = {r.mean_training_clusters:,.0f}',(r.auc,r.r_dsi),xytext=(dx,dy),textcoords='offset points',fontsize=7,ha='left',arrowprops={'arrowstyle':'-','lw':.5,'color':GRAY})
 control=data.loc['all_events'];ax.scatter(control.auc,control.r_dsi,marker='D',s=35,facecolors='white',edgecolors=NAVY,lw=1,zorder=3)
 ax.annotate('All events',(control.auc,control.r_dsi),xytext=(-6,10),textcoords='offset points',ha='center',fontsize=7)
 ax.set_title('(a) AUC–R-DSI',loc='left',fontweight='bold',color=NAVY);ax.set_xlabel('Pooled OOF AUC');ax.set_ylabel('R-DSI');ax.margins(x=.20,y=.17);ax.grid(alpha=.18);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.4f}'));ax.tick_params(axis='x',labelrotation=20)
 for i,(metric,title) in enumerate([('auc','(b) ΔAUC'),('reference_theta','(c) ΔReference θ'),('brier','(d) ΔBrier')]):
  a=fig.add_subplot(grid[i,1]);a.axhline(0,color=GRAY,lw=.7)
  for j,(key,color) in enumerate(zip(settings,colors)):
   r=interval.loc[interval.experiment.eq('spatial')&interval.setting.eq(key)&interval.metric.eq(metric)].iloc[0]
   err(a,j,r.difference,r.ci95_low,r.ci95_high,color)
  a.set_xticks(range(5),['400','600','800','1000','1200']);a.set_xlim(-.35,4.35);a.set_title(title,loc='left',fontweight='bold',color=NAVY);a.grid(axis='y',alpha=.15)
  if i==2:a.set_xlabel('Spatial clustering threshold (m)')
  else:a.tick_params(axis='x',labelbottom=False)
 handles=[Line2D([],[],marker='o',linestyle='',markerfacecolor=ORANGE,markeredgecolor=RED,label='Pareto nondominated'),Line2D([],[],marker='D',linestyle='',markerfacecolor='white',markeredgecolor=NAVY,label='All-event control')]
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,frameon=False,fontsize=7)
 fig.text(.5,.018,'Point area: mean training-cluster count G. Bars: 95% paired-seed bootstrap intervals.',ha='center',fontsize=6.7,color=NAVY)
 save(fig,'Fig8_corrected',jpeg=True)

def fig9():
 data=read(ROOT/'tables/Fig9_plot_data.csv').set_index('setting');interval=read(ROOT/'tables/sweep_paired_intervals.csv');seed=read(ROOT/'tables/sweep_seed_results.csv')
 fig,axes=plt.subplots(2,2,figsize=(7.2,4.6));fig.subplots_adjust(left=.10,right=.98,top=.91,bottom=.18,hspace=.62,wspace=.36)
 qkeys=['attr_q_0p05','cutoff_800','attr_q_0p15','attr_q_0p2'];passkeys=['pass_0p5','cutoff_800','pass_0p7','pass_0p8']
 for a,keys,x,title,color in [(axes[0,0],qkeys,[.05,.1,.15,.2],'(a) Attribute quantile',BLUE),(axes[0,1],passkeys,[.5,.6,.7,.8],'(b) Attribute pass fraction',REDORANGE)]:
  y=data.loc[keys,'mean_training_clusters'];a.plot(x,y,'o-',color=color,ms=4,lw=1);a.set_ylabel('Mean training clusters');a.set_xlabel('q' if a==axes[0,0] else 'Pass fraction');a.set_xticks(x);a.set_title(title,loc='left',fontweight='bold',color=NAVY);a.grid(alpha=.15);a.axvline(.1 if a==axes[0,0] else .6,color=RED,lw=.7,ls='--')
 keys=['cutoff_800','attr_q_0p05','attr_q_0p15','attr_q_0p2','pass_0p5','pass_0p7','pass_0p8'];labels=['Baseline','q=.05','q=.15','q=.20','pass=.50','pass=.70','pass=.80']
 for a in axes[1]:a.axhline(0,color=GRAY,lw=.7);a.set_xticks(range(7),labels,rotation=35,ha='right');a.grid(axis='y',alpha=.15)
 a=axes[1,0]
 for metric,color,sign,label in [('auc',BLUE,1,'ΔAUC'),('brier',REDORANGE,-1,'−ΔBrier')]:
  values=[]
  for j,key in enumerate(keys):
   r=interval.loc[interval.experiment.eq('attribute')&interval.setting.eq(key)&interval.metric.eq(metric)].iloc[0];lo,hi=(r.ci95_low,r.ci95_high) if sign==1 else (-r.ci95_high,-r.ci95_low);err(a,j,r.difference*sign,lo,hi,color);values.append(r.difference*sign)
  a.plot(range(7),values,color=color,lw=.8,label=label)
 a.set_title('(c) Predictive effects',loc='left',fontweight='bold',color=NAVY);a.legend(frameon=False,fontsize=7,loc='lower left')
 a=axes[1,1];values=[]
 for j,key in enumerate(keys):
  r=interval.loc[interval.experiment.eq('attribute')&interval.setting.eq(key)&interval.metric.eq('reference_theta')].iloc[0];err(a,j,r.difference,r.ci95_low,r.ci95_high,GREEN);values.append(r.difference)
 a.plot(range(7),values,color=GREEN,lw=.8);a.set_title('(d) ΔReference θ',loc='left',fontweight='bold',color=NAVY)
 fig.text(.5,.03,'Training-only clustering. Effects use the matched all-event control; bars are 95% paired-seed intervals.',ha='center',fontsize=6.7,color=NAVY)
 save(fig,'Fig9_corrected')

def fig10():
 data=read(ROOT/'tables/Fig10_plot_data.csv');r=data.loc[data.metric.eq('delta_auc')].pivot(index='feature',columns='mode',values='mean')
 order=r['global'].sort_values(ascending=False).index.tolist();fig,axes=plt.subplots(1,3,figsize=(7.5,4.5),gridspec_kw={'width_ratios':[1,1,.95]});fig.subplots_adjust(left=.165,right=.98,top=.91,bottom=.15,wspace=.52)
 for a,metric,title in zip(axes[:2],['delta_auc','delta_brier'],['(a) AUC decrease','(b) Brier increase']):
  for mode,offset,color,label in [('global',-.13,BLUE,'Across validation fold'),('within_5km_block',.13,REDORANGE,'Within 5-km groups')]:
   sub=data.loc[data.metric.eq(metric)&data['mode'].eq(mode)].set_index('feature').loc[order];y=np.arange(16)+offset
   a.hlines(y,sub.ci95_low,sub.ci95_high,color=color,lw=.8);a.scatter(sub['mean'],y,c=color,s=9,label=label)
  a.set_yticks(range(16),[LABELS[k] for k in order] if a==axes[0] else []);a.invert_yaxis();a.axvline(0,color=GRAY,lw=.7,ls='--');a.grid(axis='x',alpha=.15);a.set_title(title,loc='left',fontweight='bold',color=NAVY);a.set_xlabel('Permutation loss')
 a=axes[2];a.scatter(r['global'],r.within_5km_block,c=GREEN,s=12);hi=max(r.max())*1.08;a.plot([0,hi],[0,hi],color=GRAY,lw=.7,ls='--');a.set_xlim(-.002,hi);a.set_ylim(-.002,max(.035,r.within_5km_block.max()*1.2));a.grid(alpha=.15);a.set_title('(c) Scale agreement',loc='left',fontweight='bold',color=NAVY);a.set_xlabel('Across-fold ΔAUC');a.set_ylabel('Within-group ΔAUC')
 abbreviations={'precipitation_mean':'Mean precip.','precipitation_max':'Max precip.','Slope':'Slope','NDVI_mean':'Mean NDVI','Distance_to_river':'River distance'}
 for feature,label in abbreviations.items():
  x=r.loc[feature,'global'];y=r.loc[feature,'within_5km_block'];a.annotate(label,(x,y),xytext=(-3 if feature=='precipitation_mean' else 3,4),textcoords='offset points',ha='right' if feature=='precipitation_mean' else 'left',fontsize=6)
 fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.50,.028),ncol=2,frameon=False,fontsize=7)
 save(fig,'Fig10_corrected')

def fig6():
 d=read(ROOT/'tables/Fig6_plot_data.csv');seed=read(ROOT/'tables/Fig6_seed_results.csv');fig,axes=plt.subplots(2,3,figsize=(7.2,4.6));fig.subplots_adjust(left=.08,right=.98,top=.80,bottom=.15,hspace=.65,wspace=.60)
 settings=[('radius','(a) Pseudo-absence\nexclusion distance',1000,NAVY),('block','(b) Validation-block width',5000,ORANGE),('gap','(c) Train–validation separation',0,REDORANGE)]
 for col,(fam,title,ref,color) in enumerate(settings):
  values=sorted(d.loc[d.family.eq(fam),'scale_m'].unique())
  for row,metric in enumerate(['auc','brier']):
   a=axes[row,col];sub=d.loc[d.family.eq(fam)&d.metric.eq(metric)].set_index('scale_m').loc[values];y=np.arange(len(values))
   for j,scale in enumerate(values):
    r=sub.loc[scale];a.hlines(j,r.difference-r.difference_SD,r.difference+r.difference_SD,color=color,lw=2,alpha=.55);a.scatter(r.difference,j,s=13,c=color if scale!=ref else 'white',edgecolors=color,lw=.8,zorder=3)
    ss=seed.loc[seed.family.eq(fam)&seed.scale_m.eq(scale)].set_index('seed');reference=seed.loc[seed.family.eq(fam)&seed.scale_m.eq(ref)].set_index('seed');delta=ss[metric]-reference[metric]
    a.scatter(delta,np.full(5,j),s=4,c=color,alpha=.22)
   a.set_yticks(y,[f'{v/1000:g} km' for v in values]);a.invert_yaxis();a.axvline(0,color=GRAY,lw=.7);a.set_xlabel('ΔAUC' if metric=='auc' else 'ΔBrier');a.set_title('AUC' if row==0 else 'Brier (lower is better)',loc='left',fontsize=8,fontweight='bold');a.grid(axis='x',alpha=.12)
  axes[0,col].text(0,1.54,title,transform=axes[0,col].transAxes,fontsize=8,color=NAVY)
  axes[0,col].text(0,1.26,f'Baseline: {ref/1000:g} km',transform=axes[0,col].transAxes,fontsize=7,color=NAVY)
 fig.text(.5,.028,'Dots: paired seed summaries. Bars: ±1 SD across five seeds. Original observations remain uncompressed.',ha='center',fontsize=6.6,color=NAVY)
 save(fig,'Fig6_corrected',jpeg=True)

def main():
    fig8()
    fig9()
    fig10()
    if (ROOT / 'tables/Fig6_plot_data.csv').exists():
        fig6()

if __name__ == '__main__':
    main()
