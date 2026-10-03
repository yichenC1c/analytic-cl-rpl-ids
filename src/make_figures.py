"""Regenerate the three result figures used in the manuscript from saved JSON logs.

The system overview is a layout-only illustration. This script covers every plotted
quantity and uses the same dimensions, labels, aggregation, and vector-font settings as
the submitted figures.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
R=ROOT/'results'
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1.4,'legend.frameon':False,'savefig.pad_inches':.025})
SC=['random','b2w','w2b','toggle']; labels=['Random','B2W','W2B','Toggle'];colors=['#245B91','#A73857','#33764C','#7554A0'];styles=['-','--','-.',':'];markers=['o','s','^','D']
fig,ax=plt.subplots(figsize=(3.42,2.25),layout='constrained')
for sc,l,c,ls,mk in zip(SC,labels,colors,styles,markers):
 d=json.loads((R/f'E4_equivalence_{sc}_s0_D64.json').read_text())['curve'];ax.semilogy([x['t'] for x in d],[x['rel'] for x in d],label=l,color=c,ls=ls,marker=mk,markevery=6,ms=3,mew=.6,mfc='white')
ax.set(xlabel='Domains observed',ylabel='Relative weight error',xlim=(1,48),ylim=(1e-13,2e-10));ax.set_xticks([1,12,24,36,48]);ax.grid(axis='y',alpha=.25);ax.legend(ncol=2,loc='lower right');fig.savefig(OUT/'fig2_equivalence.pdf');plt.close(fig)
# Plot head plus random-map persistent state, not training workspace.
a={}
for f in sorted(R.glob('E5_dsweep_*.json')):
 for d in json.loads(f.read_text())['sweep']:a.setdefault(d['D'],[]).append(d['f1'])
Ds=sorted(a);mem=np.array([8*(d*d+2*d+10*d+d)/1024 for d in Ds]);f1=np.array([np.mean(a[d]) for d in Ds]);sd=np.array([np.std(a[d],ddof=1) for d in Ds])
fig,ax=plt.subplots(figsize=(3.42,2.35),layout='constrained')
ax.errorbar(mem,f1,yerr=sd,color=colors[0],marker='o',ms=3,capsize=2,elinewidth=.8)
j=Ds.index(64);ax.plot(mem[j],f1[j],marker='o',ms=8,mfc='none',mec=colors[1],mew=1.3,ls='none',label='Selected width 64')
for d,x,y in zip(Ds,mem,f1):ax.annotate(str(d),(x,y),xytext=(0,6 if d not in [64,512] else -13),textcoords='offset points',ha='center',fontsize=8)
ax.set(xscale='log',xlabel='Head and random-map state / KiB',ylabel=r'Mean weighted $F_1$',ylim=(.50,.70));ax.grid(axis='y',alpha=.25);ax.legend(loc='lower right');fig.savefig(OUT/'fig3_memory.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(3.42,2.25),layout='constrained')
for sc,l,c,ls,mk in zip(SC,labels,colors,styles,markers):
 vals=[]
 for f in sorted(R.glob(f'E1_ours_{sc}_s*_D64.json')):
  m=np.array(json.loads(f.read_text())['matrix_f1']);vals.append([np.mean(m[t,:t]-m.diagonal()[:t]) for t in range(1,len(m))])
 ax.plot(range(2,49),np.mean(vals,axis=0),label=l,color=c,ls=ls,marker=mk,markevery=6,ms=3,mew=.6,mfc='white')
ax.axhline(0,color='.45',lw=.6);ax.set(xlabel='Domains observed',ylabel='Backward transfer',xlim=(1,48),ylim=(-.115,.035));ax.set_xticks([1,12,24,36,48]);ax.grid(axis='y',alpha=.25);ax.legend(ncol=2,loc='lower left',columnspacing=1.4);fig.savefig(OUT/'fig4_bwt.pdf');plt.close(fig)
