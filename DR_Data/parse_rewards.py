#coding: utf-8

## Command
# ```python parse_rewards.py <scoring_function> <number of samples> <number of drugs>```

import numpy as np
import sys
import pandas as pd

assert len(sys.argv) == 4
nsamples=int(sys.argv[2])
ndrugs=int(sys.argv[3])
filen = "rewards_"+sys.argv[1]+"_"+str(ndrugs)+"drugs_"+str(nsamples)+"samples"

ranking_by = ["mean", "median"][0]
print_test = False
print_pc = False

#############
## PARSING ##
#############

with open(filen+".txt", "r") as f:
	lines = f.read().split("\n")

i = 0
rewards = [None]*(ndrugs*nsamples)
scores = [None]*(ndrugs*nsamples)
names = [None]*(ndrugs*nsamples)
sig_scores = [None]*(ndrugs*nsamples)
for line in lines:
	if ("Arm" in line):
		reward, score, sig_score = line.split(", ")[-3:]
		reward, score, sig_score = float(reward[1:]), float(score[:-2]), float(sig_score[:-2])
		rewards[i] = reward
		scores[i] = score
		sig_scores[i] = sig_score
		names[i] = line.split(", ")[2][1:-2]
		i += 1

scores = scores[:ndrugs]
names = names[:ndrugs]
sig_scores = sig_scores[:ndrugs]
rewards = np.array(rewards).reshape((nsamples, ndrugs))
np.savetxt(filen+".csv", rewards)

mean_rewards = rewards.mean(axis=0)
#np.savetxt(filen+"_mean.csv", mean_rewards)
ids_ = np.argwhere(mean_rewards != -666).T.flatten().tolist()
ndrugs = len(ids_)
mean_rewards = mean_rewards[ids_]
rewards = rewards[:,ids_]
scores = np.array(scores)[ids_].tolist()
names = np.array(names)[ids_].tolist()
sig_scores = np.array(sig_scores)[ids_].tolist()

remove_zeros = True

if (remove_zeros):
	ids_ = np.argwhere(np.array(scores) != 0).T.flatten().tolist()
	ndrugs = len(ids_)
	mean_rewards = mean_rewards[ids_]
	rewards = rewards[:,ids_]
	scores = np.array(scores)[ids_].tolist()
	names = np.array(names)[ids_].tolist()
	sig_scores = np.array(sig_scores)[ids_].tolist()

#############
## PLOTS   ##
#############

fsize = 25
msize = 20

import matplotlib.pyplot as plt
import os
import subprocess as sb

if (not os.path.exists("plots/")):
	sb.call("mkdir plots/", shell=True)

if (ranking_by == "mean"):
	ids = np.argsort(mean_rewards).tolist()
else:
	ids = np.argsort(np.median(rewards,axis=0)).tolist()
ids.reverse()
positive = [i+1 for i in range(len(ids)) if (scores[ids[i]] > 0)]
negative = [i+1 for i in range(len(ids)) if (scores[ids[i]] < 0)]

## Compute Hit rates @k baseline and our scoring
from sklearn.metrics import accuracy_score
ids_baseline = np.argsort(sig_scores).tolist()
ids_baseline.reverse()
if (remove_zeros):
	## Remove drugs labelled score=0
	ids_nonzero = [idx for idx in ids if (scores[idx] != 0)]
	ids_baseline_ = [idx for idx in ids_baseline if (scores[idx] != 0)]
	ids_ = ids_nonzero
else:
	ids_baseline_ = ids_baseline
	ids_ = ids
K = [1,min(2,len(ids_)),min(5,len(ids_)),min(10, len(ids_))]
hrs = []
for k in K:
	scs = [1]*k
	ae_k = [int(scores[i] > 0) for i in ids_[:k]]
	ae_k_baseline = [int(scores[i] > 0) for i in ids_baseline_[:k]]
	pc_k = [int(scores[i] < 0) for i in ids_[-k:]]
	pc_k_baseline = [int(scores[i] < 0) for i in ids_baseline_[-k:]]
	hrs.append([k]+list(map(lambda x : round(accuracy_score(x, scs), 2), [ae_k, ae_k_baseline, pc_k, pc_k_baseline])))
from scipy.stats import ks_2samp, spearmanr, kendalltau
from random import seed
npermutations=10000
stat, p = [None]*npermutations, [None]*npermutations
stat_baseline, p_baseline = [None]*npermutations, [None]*npermutations
method = ["ks_2samp", "spearmanr", "kendalltau"][0]
n = ndrugs
for nperm in range(npermutations):
	seed(nperm)
	ids__ = np.random.choice(ids, size=n, replace=True, p=None)
	ids_baseline__ = np.random.choice(ids_baseline, size=n, replace=True, p=None)
	## Test null hypothesis
	ls = eval(method)([float(mean_rewards.tolist()[i]) for i in ids__], [int(scores[i] > 0) for i in ids__])
	ls_baseline = eval(method)([float(sig_scores[i]) for i in ids_baseline__], [int(scores[i] > 0) for i in ids_baseline__])
	stat[nperm] = float(ls[0])
	p[nperm] = float(ls[1])
	stat_baseline[nperm] = float(ls_baseline[0])
	p_baseline[nperm] = float(ls_baseline[1])
stat, p = np.mean(stat), np.mean(p)
stat_baseline, p_baseline = np.mean(stat_baseline), np.mean(p_baseline)

## Produce plots
for nplot in range(2):
	fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(30, 30))
	medianprops = dict(linestyle='-', linewidth=2.5, color='lightcoral')
	meanpointprops = dict(marker='.', markerfacecolor='white', alpha=0.)
	ax.boxplot([rewards[:,i] for i in ids], showmeans=True, labels=[names[i] for i in ids], vert=True, meanline=False, meanprops=meanpointprops, notch=False, medianprops=medianprops)
	if (ranking_by == "mean"):
		ax.plot(range(1,ndrugs+1), [mean_rewards.tolist()[i] for i in ids], 'k-', label="mean")
	else:
		ax.plot(range(1,ndrugs+1), [np.median(rewards,axis=0).tolist()[i] for i in ids], 'k-', label="median")
	if (ranking_by == "mean"):
		ax.plot(positive, [mean_rewards.tolist()[ids[i-1]] for i in positive], 'gD', label="score > 0", markersize=msize)
		ax.plot(negative, [mean_rewards.tolist()[ids[i-1]] for i in negative], 'rD', label="score < 0", markersize=msize)
	else:
		ax.plot(positive, [np.median(rewards,axis=0).tolist()[ids[i-1]] for i in positive], 'gD', label="score > 0", markersize=msize)
		ax.plot(negative, [np.median(rewards,axis=0).tolist()[ids[i-1]] for i in negative], 'rD', label="score < 0", markersize=msize)
	#ax.legend()
	ax.set_ylabel("Reward", fontsize=fsize)
	ax.set_xlabel("Drug", fontsize=fsize)
        drug_labels = [names[idx]+"\n(mean="+str(round(mean_rewards.tolist()[idx], 3))+")" for idx in ids]
        drug_memberships = ["PC" if (i in negative) else ("AE" if (i in positive) else "") for i in range(1, ndrugs+1)]
        drug_ticks = ["\n".join(ls) for ls in zip(drug_labels, drug_memberships)]
	ax.set_xticklabels(drug_ticks, rotation=90, fontsize=fsize)
	cols=["r" if (i in negative) else ("g" if (i in positive) else "k") for i in range(1, ndrugs+1)]
	for xtick, color in zip(ax.get_xticklabels(), cols):
		xtick.set_color(color)
	if (nplot > 0):
		m, M = min(np.min(rewards),np.min(sig_scores)), max(np.max(rewards),np.max(sig_scores))
		ax.set_yticks(np.round(np.arange(m, M, 0.05), 2))
		ax.plot(positive, [sig_scores[ids[i-1]] for i in positive], 'gD', markersize=msize)
		ax.plot(negative, [sig_scores[ids[i-1]] for i in negative], 'rD', markersize=msize)
		ax.plot(range(1,ndrugs+1), [sig_scores[i] for i in ids], 'b-', label="baseline (CDS^2)")
		ax2=ax.twinx()
		ax2.plot(range(1,ndrugs+1), [sig_scores[i] for i in ids], 'w.', alpha=0.)
		ax2.set_ylabel("Baseline drug ranking", fontsize=fsize)
		ax2.set_yticks(np.linspace(np.min(sig_scores+rewards), np.max(sig_scores+rewards), num=ndrugs).tolist())
                positive2 = [i+1 for i in range(len(ids_baseline)) if (scores[ids_baseline[i]] > 0)]
                negative2 = [i+1 for i in range(len(ids_baseline)) if (scores[ids_baseline[i]] < 0)]
                cols=["r" if (i in negative2) else ("g" if (i in positive2) else "k") for i in range(1, ndrugs+1)]
                drug_labels = [names[idx]+"\n(score="+str(round(sig_scores[idx],2))+")" for idx in ids_baseline]
                drug_memberships = ["PC" if (i in negative2) else ("AE" if (i in positive2) else "") for i in range(1, ndrugs+1)]
                drug_ticks = ["\n".join(ls) for ls in zip(drug_labels, drug_memberships)]
		ax2.set_yticklabels(drug_ticks, fontsize=fsize)
		positive2 = [i+1 for i in range(len(ids_baseline)) if (scores[ids_baseline[i]] > 0)]
		negative2 = [i+1 for i in range(len(ids_baseline)) if (scores[ids_baseline[i]] < 0)]
		cols=["r" if (i in negative2) else ("g" if (i in positive2) else "k") for i in range(1, ndrugs+1)]
		for ytick, color in zip(ax2.get_yticklabels(), cols):
			ytick.set_color(color)
		if (print_pc):
			hrs_baseline_str = "AE (PC) "+reduce(lambda x,y : x+" | "+y, ["HR@"+str(hr[0])+"="+str(hr[2])+" ("+str(hr[4])+")" for hr in hrs])
		else:
			hrs_baseline_str = "AE "+reduce(lambda x,y : x+" | "+y, ["HR@"+str(hr[0])+"="+str(hr[2]) for hr in hrs])
	ax.set_yticklabels(ax.get_yticks(), fontsize=fsize)
	if (print_pc):
		hrs_str = "AE (PC) "+reduce(lambda x,y : x+" | "+y, ["HR@"+str(hr[0])+"="+str(hr[1])+" ("+str(hr[3])+")" for hr in hrs])
	else:
		hrs_str = "AE "+reduce(lambda x,y : x+" | "+y, ["HR@"+str(hr[0])+"="+str(hr[1]) for hr in hrs])
	if (print_test):
		stat_test = "  Scoring: "+method+" (npermutations="+str(npermutations)+") stat="+str(round(stat, 2))+" p="+str(round(p, 2))+((" **"+"*"*int(p <= 0.01)) if (p <= 0.05) else "")+"\n"
		if (nplot > 0):
			stat_test += "  Baseline: "+method+" (npermutations="+str(npermutations)+") stat="+str(round(stat_baseline, 2))+" p="+str(round(p_baseline, 2))+((" **"+"*"*int(p_baseline <= 0.01)) if (p_baseline <= 0.05) else "")+"\n"
	else:
		stat_test = ""
	ax.set_title("Boxplots of rewards from GRN across "+str(nsamples)+" patients\n(versus known associations/scores for "+str(ndrugs)+" drugs)\n"+stat_test+"\n Scoring: "+hrs_str+("\nBaseline: "+hrs_baseline_str if (nplot > 0) else ""), fontsize=fsize)
	plt.savefig("plots/"+filen+("_with_baseline" if (nplot > 0) else "")+".png", bbox_inches="tight")
	plt.close()



