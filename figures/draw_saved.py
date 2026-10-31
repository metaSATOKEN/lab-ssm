"""Draw the revised illustrations from saved JSON only. Run in the lab-ssm checkout."""
from pathlib import Path
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
import numpy as np
here=Path(__file__).resolve().parent; data=here.parent/'results'
def read(name): return json.loads((data/name).read_text())
def save(fig,name):
    for ext in ['pdf','png']:
        fig.savefig(here/(name+'.'+ext),dpi=190,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
blue='#00679a'; orange='#b65300'; teal='#08785c'; grey='#65717e'; purple='#7855a3'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,'pdf.fonttype':42})
# Continuous relative-noise profile: no integer bit allocation.
q=read('phase_c3b_sigma_rel_sweep.json')['sweep_results']; s=[.05,.1,.2]
f,ax=plt.subplots(figsize=(6.8,3.8),layout='constrained')
for k,name,c,marker in [('full','Joint score (full)',blue,'o'),('proxy_correct','Persistence, square root',teal,'s'),('proxy_C2','Persistence, linear',orange,'^'),('anti_full','Inverse allocation',purple,'D'),('uniform','Uniform',grey,'x')]:
    y=[q[str(x)][k]['pct_of_base'] for x in s];ax.plot(s,y,label=name,c=c,marker=marker)
ax.plot(s,[q['0.05']['full']['pct_of_base']*(x/.05)**2 for x in s],c='black',ls=':',label='Slope 2 reference')
ax.set(xscale='log',yscale='log',xlabel='Relative noise amplitude',ylabel='Increase in PPL (%)',title='Continuous noise on Mamba-2.8B (10 windows)')
ax.set_xticks(s,['0.05','0.10','0.20']);ax.minorticks_off();ax.grid(axis='y',alpha=.2);ax.legend(bbox_to_anchor=(1.01,.5),loc='center left',frameon=False,fontsize=9)
save(f,'fig02_sigma_scaling')
# Same inputs as the historical factorisation plot; actual widths are shown for every row.
a=read('phase_c16_matched_mean_sweep.json')['results']['Mamba-2.8B']['sweep'];b=read('phase_c16b_lyapunov_vs_range_factorisation.json')['results']['Mamba-2.8B']['sweep']
items=[('Uniform','uniform',a),('LAB (R + range)','half_slope_rg',a),('R only','half_slope',a),('Range only','range_only',b),('Shuffle R','shuffled_R_plus_r',b),('Shuffle range','R_plus_shuffled_r',b)]
f,axs=plt.subplots(1,2,sharey=True,figsize=(8.5,3.4),layout='constrained')
for ax,bits in zip(axs,[4,6]):
    for i,(_,k,d) in enumerate(items):
        e=d[f'b{bits}_{k}']; v=e['delta_pct']; mb=e['achieved_mean']
        ax.barh(i,v,color=blue if i==1 else grey,height=.60);ax.text(v+(.018*(260 if bits==4 else 42)),i,f'{v:.1f}% / {mb:.2f}b',va='center',fontsize=8.8)
    ax.set_xlim(0,365 if bits==4 else 62);ax.set_title(f'Nominal INT{bits}');ax.set_xlabel('Increase in PPL (%)');ax.grid(axis='x',alpha=.18)
axs[0].set_yticks(range(len(items)),[x[0] for x in items]);axs[0].invert_yaxis()
f.suptitle('Mamba-2.8B ablations: PPL change / achieved mean bits');save(f,'fig04_factorisation')
a=read('phase_c7_long_context_validation _v2.json')['sweep_results'];b=read('phase_c9_fp16_bf16_clip_ablation_v2.json')['sweep_results']
f,axs=plt.subplots(1,2,figsize=(8.5,3.65),layout='constrained')
for ax,d,title,ks in [(axs[0],a,'Two matched windows (C7)',[('full_A','LAB INT6',blue,'o'),('uniform','Uniform INT6',orange,'s')]),(axs[1],b,'Four matched windows (C9)',[('FP16_orig','FP16',grey,'s'),('BF16_orig','BF16',purple,'x'),('FP16_clip_only','Clip only',teal,'^'),('FP16_full_A','LAB INT6',blue,'o')])]:
    for k,label,c,m in ks:
        vals=[d[str(n)][k]['ppl'] for n in [8192,16384,32768]]; ax.plot(range(3),vals,color=c,marker=m,label=label,ls='--' if k in ['uniform','BF16_orig','FP16_clip_only'] else '-')
    ax.set_title(title);ax.set_xticks(range(3),['8K','16K','32K']);ax.set_xlabel('Context length');ax.set_ylabel('Perplexity');ax.set_xlim(-.1,2.1);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=8.5)
axs[0].set_ylim(8,15);axs[1].set_yscale('log');axs[1].set_ylim(7,270);axs[1].set_yticks([10,30,100,200]);axs[1].yaxis.set_major_formatter(ScalarFormatter());axs[1].set_ylabel('Perplexity (log scale)')
save(f,'fig05_long_context')
tr=read('phase_c10b_state_norm_pre_clip.json')['sweep_results']['32768']['FP32_patched_off']['trajectories']
f,axs=plt.subplots(3,1,figsize=(7.4,5.8),sharex=True,layout='constrained')
for (key,rows),col in zip(tr.items(),plt.cm.viridis(np.linspace(.05,.87,len(tr)))):
    for ax,k in zip(axs,['h_abs_p99','rho_p99','cliprate_pre']):
        ax.plot([r['t'] for r in rows],[r[k]*(100 if k=='cliprate_pre' else 1) for r in rows],c=col,lw=1.1,label='L'+key.split('.')[2])
axs[0].set_yscale('log');axs[0].set_ylabel('State |h|, p99');axs[1].set_ylabel('Transition, p99');axs[2].set_ylabel('Above range (%)')
axs[0].legend(loc='lower right',ncol=4,fontsize=7,frameon=False);axs[0].set_title('Sampled layer trajectories at 32K, intervention off (C10b)')
axs[2].set_xticks([0,8192,16384,24576,32767],['0','8K','16K','24K','32K']);axs[2].set_xlabel('Token position')
for ax in axs:ax.grid(axis='y',alpha=.18)
save(f,'fig06_state_norm_trajectory')
