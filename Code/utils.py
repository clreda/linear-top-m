# coding: utf-8

import csv
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import subprocess as sb

from constants import target_folder

if (not os.path.exists(target_folder)):
	sb.call("mkdir "+target_folder, shell=True)

####################################################
## Output redirection                             ##
####################################################

from contextlib import contextmanager
import io
import sys

@contextmanager
def stdout_redirector(stream):
	old_stdout = sys.stdout
	sys.stdout = stream
	try:
		yield
	finally:
		sys.stdout = old_stdout

def capture_output(f):
	i = io.BytesIO()
	with stdout_redirector(i):
		res = f(0)
	return res

#https://stackoverflow.com/questions/5136611/capture-stdout-from-a-script
from IPython.utils.capture import capture_output as capture

####################################################
## Typing                                         ##
####################################################

#' @param obj Python object
#' @param t character string for expected type
#' @return Python boolean: true if @obj is of the expected type @t
def is_of_type(obj, t):
	'''Returns a boolean: if @obj is of Python type @t'''
	generic_problems = ["generic", "bernouilli", "gaussian", "epilepsy", "epilepsySubset", "poisson", "exponential"]
	check = "class" if (t in ["numpy.matrix", "pandas.core.frame.DataFrame"]+generic_problems) else "type"
	t = "utils."+t if (t in generic_problems) else t
	if (t == "numpy.matrix" and str(type(obj))[3+len(check):-2] == "numpy.matrixlib.defmatrix.matrix"):
		t = "numpy.matrixlib.defmatrix.matrix"
	assert str(type(t)) == "<type 'str'>"
	return str(type(obj)) == "<"+check+" '"+t+"'>"

#' @param obj Python object
#' @param t character string for expected type
#' @return Python boolean: true if @obj is of the expected type @t
def is_of_type_OPTION(obj, t):
	'''Returns a boolean: if @obj is of Python type @t or is None'''
	return is_of_type(obj, t) or is_of_type(obj, "NoneType")

#' @param objs Python object list
#' @param t character string for expected type
#' @return Python boolean: true if each object in @objs is of the expected type @t
def is_of_type_LIST(objs, t):
	'''Returns a boolean: if each object in list @objs is of Python type @t'''
	return is_of_type(objs, "list") and (all([is_of_type(x, t) for x in objs]) or len(objs) == 0)

####################################################
## Feature & Score selection                      ##
####################################################

## apply binary mask on binary vector
#' @param vector binary DF
#' @param mask binary DF
#' @return result Data Frame of the masked vector
def apply_mask(vector, mask):
	masked = vector.join(mask, how="outer")[[mask.columns[0]]]
	replaces = list(filter(lambda x : x in vector.dropna().index, masked.index[pd.isnull(masked).any(1).nonzero()[0]]))
	for row in replaces:
		masked.loc[row] = vector.loc[row][vector.columns[0]]
	masked.columns = ["masked"]
	return masked

## reward file parser
def get_drug_id(drug_ls, fname):
	arm_position = 0
	total = int(fname.split("drugs")[0].split("_")[-1])
	drug_ids = [None]*len(drug_ls)
	drug_positions = [None]*len(drug_ls)
	with open(fname, "r") as f:
		lines = f.read().split("\n")
	for line in lines:
		if ("Arm" in line):
			ls = line.split(", ")
			drug_name = ls[2][1:-2]
			if (drug_name in drug_ls):
				drug_ids[drug_ls.index(drug_name)] = int(ls[1][1:])
				drug_positions[drug_ls.index(drug_name)] = arm_position
			arm_position += 1
		if (arm_position >= total):
			break
	return drug_ids, drug_positions

## Parsing the value of omega
def parse_omega_str(omega):
	omega_str = ""
	if ("pi" in omega):
		p1, p2 = omega.split("/")
		if ("/" in omega):
			omega_str = p1+p2
			if (len(p1.split("pi")[0]) == 0):
				omega = float(np.pi/float(eval(p2)))
			else:
				omega = float(eval(p1.split("pi")[0])*np.pi/float(eval(p2)))
		else:
			if (len(p1.split("pi")[0]) == 0):
				omega = float(eval(p1.split("pi")[0])*np.pi)
			else:
				omega = float(np.pi)
			omega_str = p1
	else:
		omega_str = omega.split("[.]")
		if (len(omega_str) > 1):
			omega_str = omega_str[0]+"-"+omega_str[1]
		else:
			omega_str = omega_str[0]
		omega = float(omega)
	assert is_of_type(omega, "float") and is_of_type(omega_str, "str")
	assert omega > 0
	return omega, omega_str

#' @param output Python character string
#' @param t Python integer
#' @param genes Python list of character strings
#' @param solmax Python integer
#' @param ch Python integer
#' @return df Pandas DataFrame of state (#genes rows x 1 column)
def get_state(output, t, genes, solmax=0, ch=None):
	select_trajectory = lambda state, ch : state[(ch*len(genes)):((ch+1)*len(genes))]
	to_df = lambda state : pd.DataFrame(state, index=genes, columns=["state"])
	lines = list(filter(lambda x : "step"+str(t)+" " in x, output.split("\n")))
	state = "".join("".join("".join(lines).split("step"+str(t)+" ")).split(" "))
	state = list(map(int, state))
	if (str(ch) == "None"):
		## Select a trajectory at random
		if (solmax == 0):
			solmax = int(len(state)/float(len(genes)))
		ch = np.random.choice(range(solmax), p=[1/float(solmax)]*solmax)
		state = select_trajectory(state, ch)
		assert len(state) == len(genes)
		return to_df(state), ch
	state = select_trajectory(state, ch)
	assert len(state) == len(genes)
	return to_df(state)

## SOURCE adapted from https://www.sthu.org/blog/13-perstopology-peakdetection/index.html
def get_persistent_homology(seq):
    ## Keep track of time of birth and time of death for each peak
    class Peak:
        def __init__(self, startidx):
            self.born = self.left = self.right = startidx
            self.died = None

        def get_persistence(self, seq):
            return float("inf") if self.died is None else seq[self.born] - seq[self.died]
    peaks = []
    # Maps indices to peaks
    idxtopeak = [None for s in seq]
    # Sequence indices sorted by decreasing values
    indices = range(len(seq))
    indices = sorted(indices, key = lambda i: seq[i], reverse=True)
    # Process each sample in descending order
    for idx in indices:
        lftdone = (idx > 0 and idxtopeak[idx-1] is not None)
        rgtdone = (idx < len(seq)-1 and idxtopeak[idx+1] is not None)
        il = idxtopeak[idx-1] if lftdone else None
        ir = idxtopeak[idx+1] if rgtdone else None
        # New peak born
        if not lftdone and not rgtdone:
            peaks.append(Peak(idx))
            idxtopeak[idx] = len(peaks)-1
        # Directly merge to next peak left
        if lftdone and not rgtdone:
            peaks[il].right += 1
            idxtopeak[idx] = il
        # Directly merge to next peak right
        if not lftdone and rgtdone:
            peaks[ir].left -= 1
            idxtopeak[idx] = ir
        # Merge left and right peaks
        if lftdone and rgtdone:
            # Left was born earlier: merge right to left
            if seq[peaks[il].born] > seq[peaks[ir].born]:
                peaks[ir].died = idx
                peaks[il].right = peaks[ir].right
                idxtopeak[peaks[il].right] = idxtopeak[idx] = il
            else:
                peaks[il].died = idx
                peaks[ir].left = peaks[il].left
                idxtopeak[peaks[ir].left] = idxtopeak[idx] = ir
    # This is optional convenience
    return sorted(peaks, key=lambda p: p.get_persistence(seq), reverse=True)

#' @param df Pandas DataFrame of non-binary signatures (from same plate, same condition -> technical/biological replicates)
#' @param thres nbins in histogram
#' @return binary_sig binary (UP/DOWN 1/0 regulated  NA otherwise) signature for treated
def binarize_via_histogram(df, samples=[], thres=20, full=False, min_genes=5, bkgrnd_thres_upregulated=None, bkgrnd_thres_downregulated=None):
	df = df.dropna()
	genes = list(df.index)
	counts_, thresholds_ = np.histogram(df.values.flatten(), bins=thres)
	## We don't take into account bins with less than 5 genes (background noise)
	keep_ids = list(filter(lambda i : counts_[i] > min_genes, range(len(counts_))))
	counts, thresholds = counts_[keep_ids], thresholds_[keep_ids]
	if (any([str(t) == "None" for t in [bkgrnd_thres_upregulated, bkgrnd_thres_downregulated]])):
		## algorithm using persistent homology to find peaks and their significance
		## https://www.sthu.org/blog/13-perstopology-peakdetection/index.html
		## returned peaks are already ranked by persistence/"significance" value
		ids = [peak.born for peak in get_persistent_homology(counts)]
		assert len(ids) > 0
		ids = sorted(ids[:2], key=lambda i : thresholds[i])
		if (len(ids) == 1):
			bkgrnd_thres_upregulated = thresholds[min(ids[0]+1, len(thresholds)-1)]
			## the +1 is important as we want to trim the genes in the corresponding bin
			bkgrnd_thres_downregulated = thresholds[min(max(ids[0]-1, 0)+1, len(thresholds)-1)]
			if (bkgrnd_thres_upregulated == bkgrnd_thres_downregulated):
				bkgrnd_thres_upregulated += np.sqrt(np.var(thresholds_))
		else:		
			bkgrnd_thres_upregulated = thresholds[ids[-1]]
			## the +1 is important as we want to trim the genes in the corresponding bin
			bkgrnd_thres_downregulated = thresholds[ids[0]+1]	
	assert bkgrnd_thres_downregulated < bkgrnd_thres_upregulated
	## Simple binarization
	signature = [None]*len(genes)
	for sidx, idx in enumerate(genes):
		if (np.mean(df.loc[idx].values) < bkgrnd_thres_downregulated):
			signature[sidx] = 0
		else:
			if (np.mean(df.loc[idx].values) > bkgrnd_thres_upregulated):
				signature[sidx] = 1
			else:
				signature[sidx] = np.nan
	binary_sig = pd.DataFrame(signature, index=genes, columns=["aggregated"])
	if (full):
		ids_ = [thresholds_.tolist().index(thresholds[i]) for i in ids]
		return [binary_sig, [bkgrnd_thres_downregulated, bkgrnd_thres_upregulated, thresholds_, counts_, ids_]]
	return binary_sig

#' @param observations_save_fname Python character string
#' @param nsteps Python integer
#' @param perts list of gene perturbations
#' @param genes Python character string list
#' @param experiments Python dictionary list (1 dictionary per experiment: keys: "cell", "dose", "ptype", "pert", "itime", "exprs")
#' @param verbose Python integer
#' @return integer: number of written experiments (write formatted experiments to file)
def write_observations_file(observations_save_fname, nsteps, perts, genes, experiments, verbose=0, save_path="", ignore=True, cell_spe_nsteps=3):
	assert all([all([k in list(exp.keys()) for k in ["cell", "dose", "ptype", "pert", "itime", "exprs"]]) for _, exp in enumerate(experiments)])
	## Create observations file
	obs_str = ""
	sig_str = ""
	perturbed_ko = list(filter(lambda x : "-" in perts[genes.index(x)], genes))
	perturbed_fe = list(filter(lambda x : "+" in perts[genes.index(x)], genes))
	ko_exists = len(perturbed_ko) > 0
	fe_exists = len(perturbed_fe) > 0
	cells = list(set([str(exp["cell"]) for _, exp in enumerate(experiments)]))
	chrom_path = False
	exp_id = 0
	for _, exp in enumerate(experiments):
		closed_genes = []
		if (chrom_path and ignore):
			if (len(closed_genes) == 0):
				continue
		pert = exp["pert"]
		if ("perturbed_oe" in list(exp.keys())):
			perturbed_oe_ = list(set(([pert] if ("trt_oe" in exp["ptype"]) else [])+exp["perturbed_oe"]))
		else:
			perturbed_oe_ = [pert] if ("trt_oe" in exp["ptype"]) else []
		if ("perturbed_ko" in list(exp.keys())):
			perturbed_ko_ = list(set(([pert] if ("trt_oe" not in exp["ptype"]) else [])+exp["perturbed_ko"]))
		else:
			perturbed_ko_ = [pert] if ("trt_oe" not in exp["ptype"]) else []		
		desc = "cell "+exp["cell"]+"; "+exp["itime"]+"; dose "+exp["dose"]+"; perturbagen "+pert+" ("+exp["ptype"]+")"
		obs_str += "// " + desc + "\n"
		pert_title = "KnockDown" if ("trt_oe" not in exp["ptype"]) else "OverExpression"
		if (pert_title == "OverExpression" and ko_exists):
			obs_str += "#Experiment"+str(exp_id+1)+"[0] |= $KnockDown"+str(exp_id+1)+";\n"
		if (pert_title == "KnockDown" and fe_exists):
			obs_str += "#Experiment"+str(exp_id+1)+"[0] |= $OverExpression"+str(exp_id+1)+";\n"
		obs_str += "#Experiment"+str(exp_id+1)+"[0] |= $"+pert_title+str(exp_id+1)+";\n"
		if ("Initial" in list(exp["exprs"].keys())):
			k = "Initial"
			obs_str += "#Experiment"+str(exp_id+1)+"["+str(exp["exprs"][k]["step"])+"] |= $"+str(k)+str(exp_id+1)+";\n"
		for k in list(exp["exprs"].keys()):
			if (k not in ["Final", "Initial"]):
				obs_str += "#Experiment"+str(exp_id+1)+"["+str(exp["exprs"][k]["step"])+"] |= $"+str(k)+str(exp_id+1)+";\n"
		if ("Final" in list(exp["exprs"].keys())):
			obs_str += "#Experiment"+str(exp_id+1)+"["+str(exp["exprs"][k]["step"])+"] |= $Final"+str(exp_id+1)+";\n"
			obs_str += "#Experiment"+str(exp_id+1)+"["+str(exp["exprs"][k]["step"]+1)+"] |= $Final"+str(exp_id+1)+";\n"
			obs_str += "fixpoint(#Experiment"+str(exp_id+1)+"["+str(exp["exprs"][k]["step"]+1)+"]);\n"
		obs_str += "\n"
		gene_fe = lambda g, perturbed_ : any([pt == g for pt in perturbed_])
		if (pert_title == "OverExpression"):
			actual_vals = [int(gene_fe(g, perturbed_oe_)) for g in perturbed_fe]
			fe_s = reduce(lambda x,y: x+" and"+y, [" FE("+g+") = "+str(actual_vals[sg]) for sg, g in enumerate(perturbed_fe)])
			sig_str += "$"+pert_title+str(exp_id+1)+" :=\n{"+fe_s+"\n};\n\n"
			if (ko_exists):
				actual_vals = [int((chrom_path and g in closed_genes) or gene_fe(g, perturbed_ko_)) for g in perturbed_ko]
				ko_s = reduce(lambda x,y: x+" and"+y, [" KO("+g+") = "+str(actual_vals[sg]) for sg, g in enumerate(perturbed_ko)])
				sig_str += "$KnockDown"+str(exp_id+1)+" :=\n{"+ko_s+"\n};\n\n"
		if (pert_title == "KnockDown"):
			actual_vals = [int(gene_fe(g, perturbed_ko_) or (chrom_path and g in closed_genes)) for g in perturbed_ko]
			ko_s = reduce(lambda x,y: x+" and"+y, [" KO("+g+") = "+str(actual_vals[sg]) for sg, g in enumerate(perturbed_ko)])
			sig_str += "$"+pert_title+str(exp_id+1)+" :=\n{"+ko_s+"\n};\n\n"
			if (fe_exists):
				actual_vals = [int(gene_fe(g, perturbed_oe_)) for g in perturbed_fe]
				fe_s = reduce(lambda x,y: x+" and"+y, [" FE("+g+") = "+str(actual_vals[sg]) for sg, g in enumerate(perturbed_fe)])
				sig_str += "$OverExpression"+str(exp_id+1)+" :=\n{"+fe_s+"\n};\n\n"
		for k in list(exp["exprs"].keys()):
			sig = exp["exprs"][k]["sig"]
			degs = list(sorted(sig.dropna().index))
			sig_s = reduce(lambda x,y: x+" and"+y, [" "+g+" = "+str(int(int(sig.loc[g]) > 0 and (not chrom_path or not g in closed_genes))) for g in degs])
			sig_str += "$"+str(k)+str(exp_id+1)+" :=\n{"+sig_s+"\n};\n\n"
		exp_id += 1
	with open(observations_save_fname, "w+") as f:
		f.write((obs_str+"\n"+sig_str).encode('utf8'))
	return exp_id+1

####################################################
## Plotting                                       ##
####################################################

#' @param sample_complexity Python integer list of size n
#' @param theoretical_bound OPTIONAL Python integer
#' @param labels OPTIONAL Python list of character strings
#' @param xtitle OPTIONAL Python character string
#' @return None
def plot_sample_complexity(sample_complexity, theoretical_bound=None, labels=None, xtitle=""):
	'''Plots sample complexity histogram across all simulations, with the theoretical bound derived from the paper on UGapE [Gabillon et al., 2012]'''
	assert is_of_type_LIST(sample_complexity, "int")
	assert is_of_type_OPTION(theoretical_bound, "int")
	assert is_of_type_OPTION(labels, "list")
	if (is_of_type(labels, "list")):
		assert is_of_type_LIST(labels, "str")
	assert is_of_type(xtitle, "str")
	n_simu = len(sample_complexity)
	plt.hist(sample_complexity, bins=100, density=False, label="# samples")
	if (is_of_type(labels, "NoneType")):
		ticks = [np.min(sample_complexity)-1]+list(sample_complexity)+[np.max(sample_complexity)+1]
	else:
		ticks = labels
	if (theoretical_bound):
		s1, s2 = theoretical_bound, np.mean(sample_complexity)
		ratio = max(s1, s2)/min(s1, s2)
		s_max = np.max(np.histogram(sample_complexity, bins=100, density=False)[0])
		y_interval = range(s_max) + [s_max]*(n_simu-s_max+1) + [s_max+1]
		if (ratio < 2):
			plt.plot([theoretical_bound]*(n_simu+2), y_interval, "r--", label="Expected SC")
			plt.text(theoretical_bound, s_max+1, "O("+str(theoretical_bound)+")", color="red")
			ticks += [theoretical_bound] if (theoretical_bound) else []
		else:
			if (s1 > s2):
				plt.plot([np.max(sample_complexity)+1]*(n_simu+2), y_interval, "r--", label="Expected SC")
				plt.text(np.max(sample_complexity), s_max+1, "O("+str(theoretical_bound)+")", color="red")
			else:
				plt.plot([np.min(sample_complexity)-1]*(n_simu+2), y_interval, "r--", label="Expected SC")
				plt.text(np.min(sample_complexity)-1, s_max+1, "O("+str(theoretical_bound)+")", color="red")
	plt.xticks(ticks)
	m = int(np.mean(sample_complexity))
	s = round(np.sqrt(np.var(sample_complexity)), 2)
	plt.xlabel('Empirical sample complexity'+xtitle)
	plt.ylabel("# samples")
	plt.title("Empirical sample complexity " +'(avg = '+str(m)+' +- '+str(s)+')')
	plt.legend(bbox_to_anchor=(0, 1), loc='best', ncol=1)

#' @param is_good_answer Python integer list of size n (only 0-1 values)
#' @param xtitle OPTIONAL Python character string
#' @return None
def plot_performance(is_good_answer, xtitle=None):
	'''Plots good identification pie chart'''
	assert is_of_type_LIST(is_good_answer, "int")
	assert all(list(map(lambda x : x in [0,1], is_good_answer)))
	assert is_of_type_OPTION(xtitle, "str")
	n_simu = len(is_good_answer)
	ga = sum(is_good_answer)
	plt.pie([ga, len(is_good_answer)-ga], labels=["Correct", "Incorrect"], colors=["green", "red"], autopct='%1.1f%%', shadow=False)
	plt.axis("equal")
	plt.title("Correct identification events "+(xtitle if (xtitle) else ""))
	plt.legend(bbox_to_anchor=(0, 1), loc='best', ncol=1)

#' @param values Python float list of size n
#' @param parameter_name character string
#' @param parameter_values Python float or integer list of size n
#' @param milestones OPTIONAL Python list of float or int values
#' @param xtitle OPTIONAL Python character string
#' @param col OPTIONAL Python character string
#' @param ms_dir OPTIONAL Python character string: either "sup" or "inf"
#' @return None
def plot_function(values, parameter_name, parameter_values, milestones=[], xtitle="", col="b", ms_dir="sup"):
	'''Plots curve of value in @values in function of parameter @parameter_name (values in @values are average values across several simulations)'''
	assert is_of_type_LIST(values, "float")
	assert is_of_type(parameter_name, "str")
	assert is_of_type_LIST(parameter_values, "float") or is_of_type_LIST(parameter_values, "int") 
	assert len(values) == len(parameter_values)
	assert is_of_type_LIST(milestones, "float") or is_of_type_LIST(milestones, "int")
	assert is_of_type(xtitle, "str")
	assert is_of_type(col, "str")
	assert is_of_type(ms_dir, "str") and ms_dir in ["sup", "inf"]
	n_values = len(values)
	plt.plot(parameter_values, values, col)
	plt.xticks(parameter_values)
	plt.xlabel('Values of parameter '+parameter_name)
	plt.ylabel("Values")
	if (len(milestones) > 0):
		sorted_values_ids = np.argsort(values).tolist()
		for m in milestones:
			idx = list(filter(lambda i : values[i] >= m if (ms_dir == "sup") else values[i] <= m, sorted_values_ids))
			if (len(idx) > 0 and len(idx) < n_values):
				idx = idx[0] if (ms_dir == "sup") else idx[-1]
				plt.plot([parameter_values[idx]], [values[idx]], "g*")
				plt.plot([parameter_values[idx]]*2, [0,values[idx]], "g--")
				plt.text(parameter_values[idx], values[idx], (">" if (ms_dir == "sup") else "<")+"= "+str(m), bbox=dict(facecolor='black', alpha=0.5))
				plt.plot([parameter_values[idx]]*2, [0,values[idx]], "g--")
	plt.title(xtitle)
	plt.legend(bbox_to_anchor=(0, 1), loc='best', ncol=1)

#' @param m Python integer
#' @param epsilon Python float
#' @param candidates Python int list
#' @param confidence_intervals NumPy narray of size (K, 2)
#' @param means NumPy narray of size K
#' @param na NumPy narray of size K
#' @param name Python character string
#' @param B Python float or int
#' @param oracle OPTIONAL Python float list of length K
#' @param best Python int
#' @param challenger Python int
#' @param indices Python float list of size K
#' @param mean_bound Python float
#' @return None
def plot_confidence_intervals(m, epsilon, candidates, confidence_intervals, means, na, name, B, oracle=None, best_arm=None, challenger=None, indices=None, mean_bound=None):
	'''Plots current confidence intervals in @confidence_intervals for the (@m, @epsilon)-best arm identification problem, show currently sampled arms in @candidates, plots arm means @means, and #samplings for each arm in @na, in algorithm @name, and shows oracle arms (if exists) using oracle scores for all arms stored in @oracle'''
	assert is_of_type(m, "int") and is_of_type(epsilon, "float")
	assert is_of_type_LIST(candidates, "int")
	assert is_of_type_OPTION(confidence_intervals, "numpy.ndarray")
	if (not is_of_type(confidence_intervals, "NoneType")):
		assert np.shape(confidence_intervals)[1] == 2
		K, _ = np.shape(confidence_intervals)
	else:
		K = means.size
	assert is_of_type(means, "numpy.ndarray") and means.size == K
	assert is_of_type(na, "numpy.ndarray") and na.size == K
	assert is_of_type(name, "str")
	assert is_of_type(B, "float") or is_of_type(B, "int")
	assert is_of_type_OPTION(oracle, "list")
	if (not is_of_type(oracle, "NoneType")):
		assert is_of_type_LIST(oracle, "float")
		assert len(oracle) == K
	assert is_of_type_OPTION(best_arm, "int")
	if (not is_of_type(best_arm, "NoneType")):
		assert best_arm < K and best_arm >= 0
	assert is_of_type_OPTION(challenger, "int")
	if (not is_of_type(challenger, "NoneType")):
		assert challenger < K and challenger >= 0
	assert is_of_type_OPTION(indices, "list")
	if (not is_of_type(indices, "NoneType")):
		assert is_of_type_LIST(indices, "float")
		assert len(indices) == K
	assert is_of_type_OPTION(mean_bound, "float")
	fig, [ax1, ax2] = plt.subplots(2,1, gridspec_kw = {'height_ratios':[4, 1]}, figsize=(20, 10))
	K = len(na)
	ids = np.argsort(means)
	sorted_means = np.sort(means)
	best = [ids[-m:], sorted_means[-m:]]
	worst = [ids[:K-m], sorted_means[:K-m]]
	t = int(sum(na))
	title = "Round in "+name+" t = "+str(t)
	if (B > epsilon):
		assert is_of_type(B, "float")
		title += " (B(t) = "+str(round(B, 3))+" > "+str(epsilon)+")"
	else:
		assert is_of_type(B, "float")
		title += " (stopped: B(t) = "+str(round(B, 3))+" <= "+str(epsilon)+")"
	ax1.set_title(title)
	## Number of times arms have been sampled so far
	s = [((min(n, 300)+1)*20) for n in na]
	ax2.scatter(range(K), [0]*K, s=s, c="blue")
	ax2.scatter(range(K), [0]*K, s=[20]*K, c="blue", label="#samples")
	ax2.legend(bbox_to_anchor=(0, 1), loc='best', ncol=1)
	for i in range(K):
		ax2.text(i, 0, str(int(na[i])), color="red")
	ax2.set_yticks([],[])
	ax2.set_xticks(range(K))
	ax2.set_xlabel("Arms")
	ax2.set_ylabel("")
	ax2.legend(bbox_to_anchor=(0, 1), loc='best', ncol=1)
	## Means
	#ax1.plot(best[0], best[1], "go", label="Estimated best arms")
	#ax1.plot(worst[0], worst[1], "ro", label="Estimated worst arms")
	ax1.plot(best[0], best[1], "ko", label="Estimated means")
	ax1.plot(worst[0], worst[1], "ko")
	if (not is_of_type(best_arm, "NoneType") and not is_of_type(challenger, "NoneType")):
		assert is_of_type(best_arm, "int") and is_of_type(challenger, "int")
		#ax1.plot(best_arm, oracle[best_arm]/float(2)+0.001, "b^", label="Best arm")
		#ax1.plot(challenger, oracle[challenger]/float(2)+0.001, "r^", label="Challenger")
		candidates = [best_arm, challenger]
		ax1.plot([i-0.07 for i in candidates], [means[i] for i in candidates], "r^", label="best arm/challenger")
	if (not is_of_type(oracle, "NoneType") and np.var(means) > 0):
		circle_true_best_arm = lambda arm : plt.Circle((arm, means[arm]), radius=0.05, color="k", fill=False)
		true_best_arms = np.argsort(oracle)[-m:]
		for arm in true_best_arms:
			ax1.add_artist(circle_true_best_arm(arm))
	if (not is_of_type(oracle, 'NoneType')):
		true_best_arms = np.argsort(oracle)[-m:]
		true_worst_arms = np.argsort(oracle)[:(K-m)]
		#ax1.plot(true_best_arms[0], oracle[true_best_arms[0]], "bo", label="True best means")
		#for arm in true_best_arms[1:]:
		#	ax1.plot(arm, oracle[arm], "bo")
		#ax1.plot(true_worst_arms[0], oracle[true_worst_arms[0]], "ko", label="True worst means")
		#for arm in true_worst_arms[1:]:
		#	ax1.plot(arm, oracle[arm], "ko")
		ax1.plot(true_best_arms[0], oracle[true_best_arms[0]], "go", label="True means")
		for arm in true_best_arms[1:]:
			ax1.plot(arm, oracle[arm], "go")
		ax1.plot(true_worst_arms[0], oracle[true_worst_arms[0]], "go")
		for arm in true_worst_arms[1:]:
			ax1.plot(arm, oracle[arm], "go")
	if (not is_of_type(confidence_intervals, "NoneType")):
		## Confidence intervals
		for i in range(K):
			ax1.plot([i]*2, confidence_intervals[i, :], "k-")
			ax1.plot([i]*2, confidence_intervals[i, :], "k+")
		## Difference between h_arm and l_arm
		if (len(candidates) > 1):
			h_arm, l_arm = candidates[0], candidates[1]
			shift_h = 0.05*(-1 if (h_arm < l_arm) else 1)+(-0.05 if (h_arm < l_arm) else 0)
			shift_l = 0.05*(1 if (h_arm < l_arm) else -1)+(0 if (h_arm < l_arm) else -0.05)
			ax1.plot([h_arm+shift_h], min([confidence_intervals[h_arm, 1]], 2*np.max(means)), "b*", label="Sampled arm")
			ax1.plot([l_arm+shift_l], min([confidence_intervals[l_arm, 1]], 2*np.max(means)), "b*")
			ax1.text(h_arm+shift_h, min(confidence_intervals[h_arm, 1], 2*np.max(means)), "h(t)", color='b')
			ax1.text(l_arm+shift_l, min(confidence_intervals[l_arm, 1], 2*np.max(means)), "l(t)", color='b')
			ax1.plot([h_arm, l_arm], [confidence_intervals[h_arm, 0]]*2, "g--")
			ax1.plot([h_arm, l_arm], [min(confidence_intervals[l_arm, 1], 2*np.max(means))]*2, "r--")
			if (epsilon > 0):
				cih_l = confidence_intervals[h_arm, 0]
				pos_eps = min(h_arm, l_arm)+0.5
				ax1.text(pos_eps+0.1, cih_l+epsilon, "eps")
				ax1.plot([pos_eps]*2, [cih_l, cih_l+epsilon], "b-")
				ax1.plot([pos_eps]*2, [cih_l, cih_l+epsilon], "b+")
		else:
			ax1.plot([i+0.07 for i in candidates], [confidence_intervals[i, 1] for i in candidates], "b*", label="Sampled arm")
	else:
		if (candidates):
			ax1.plot([i+0.07 for i in candidates], [means[i] for i in candidates], "b*", label="Sampled arm")
	#ax1.set_yticks([np.min(means)]+list(map(int, means))+[np.max(means)])
	if (not is_of_type(indices, "NoneType")):
		indices_in_plot = list(filter(lambda x : indices[x] <= np.max(means), range(len(indices))))
		ticked = False
		for i in range(len(means)):
			if (not i in indices_in_plot):
				if (not ticked):
					ax1.plot(i, np.max(means)+0.1, "rD", label="Index values")
				else:
					ax1.plot(i, np.max(means)+0.1, "rD", label="Index values")
				ax1.text(i+0.1, np.max(means)+0.1, str(round(indices[i], 2)), color='r')
				ticked = True
		ax1.plot(indices_in_plot, [indices[i] for i in indices_in_plot], "rD")
	ax1.set_yticks(means)
	ax1.set_xticks(range(K))
	## For bounded rewards
	if (not is_of_type(mean_bound, "NoneType")):
		ax1.set_ylim((-5*mean_bound, 5*mean_bound))
	ax1.set_xlabel("Arms")
	ax1.set_ylabel("Reward")
	ax1.legend(bbox_to_anchor=(0, 1), loc='best', ncol=1)

## source: adapted from https://stackoverflow.com/questions/7941226/how-to-add-line-based-on-slope-and-intercept-in-matplotlib
#' @param x Python float list
#' @param y Python float list
#' @param slope NumPy NDarray
#' @param intercept NumPy NDarray
#' @param color Python string
#' @return None
def abline(x, y, slope, intercept, color="b"):
	"""Plot a line from slope and intercept"""
	assert is_of_type_LIST(x, "float")
	assert is_of_type_LIST(y, "float")
	assert is_of_type(slope, "numpy.ndarray")
	assert is_of_type(slope, "numpy.ndarray")
	assert is_of_type(intercept, "numpy.ndarray")
	assert slope.size == intercept.size
	axes = plt.gca()
	for i in range(slope.size):
		x_vals = np.linspace(0, x[i], num=100) #np.array(axes.get_xlim())
		if (slope[i] == np.inf):
			y_vals = np.linspace(0, y[i], num=100) #np.array(axes.get_xlim())
			x_vals = np.array([intercept[i]]*y_vals.size)
		elif (slope[i] == 0):
			y_vals = np.array([intercept[i]]*x_vals.size)
		else:
			if (slope[i] < 0):
				x_vals = np.linspace(x[i], 0, num=100) #np.array(-axes.get_xlim())
			y_vals = intercept[i] + slope[i] * x_vals
		plt.plot(x_vals, y_vals, color=color)

#' @param bandit ExploreMBandit instance
#' @param ndim OPTIONAL number of dimensions (1 or 2 or 3)
#' @param style OPTIONAL Python character string ("PCA" or "TSNE")
#' @return None
def plot_instance(bandit, ndim=2, style=["PCA", "TSNE"][0]):
	'''Plots a mapping of feature vectors in @X in the bandit instance in @ndim D (grouped by @scores values (colours) and by membership to the set of best arms) using visualization method @style'''
	assert is_of_type(ndim, "int") and ndim in range(1, 4)
	assert is_of_type(style, "str") and style in ["PCA", "TSNE"]
	from sklearn.decomposition import PCA
	from sklearn.manifold import TSNE
	from mpl_toolkits.mplot3d import Axes3D
	import seaborn as sns
	import time
	import matplotlib.cm as cm
	from glob import glob
	path = bandit.path_to_plots
	if (len(glob(path+"bandit_instance*.png")) > 0):
		return None
	X = bandit.X
	scores = bandit.problem.oracle
	m = bandit.m
	title = bandit.params
	assert is_of_type(X, "numpy.matrix")
	K = np.shape(X)[1]
	assert is_of_type_LIST(scores, "float")
	assert len(scores) == K
	assert is_of_type(m, "int")
	assert is_of_type(title, "str")
	assert is_of_type(ndim, "int")
	assert ndim in range(1, 4)
	d = np.shape(X)[0]
	assert d > 0
	flatten_to_list = lambda arr : [float(sa) for sa in np.array(arr).flatten()]
	## Discretize scores: 1 <=> best m-subset (wrt scores), 2 <=> worst K-m arms
	sorted_arms = np.array(m_maximal(scores, len(scores)))
	y = np.zeros((len(scores),))+2
	chunk_nb = 2
	y[sorted_arms[-m:]] = 1
	assert sum([x == 1 for x in y]) == m
	assert sum([x == 2 for x in y]) == K-m
	if (d < 3):
		plt.figure(figsize=(16,10))
		plt.grid()
		y1 = int(np.argmax(y))
		y2 = int(np.argmin(y))
		colours = np.array([[0, 1, 0.5, 1],[1, 0, 0, 1]])
		if (d == 1):
			for i in range(chunk_nb):
				ids = y == i+1
				ids[[y1,y2]] = False
				plt.plot([0]*np.sum(ids), X[0,ids], color=colours[i], marker="*" if (i == 1) else "o")
			plt.plot([0], X[0,y1], color=colours[1], marker="o", label="worst arms")
			plt.plot([0], X[0,y2], color=colours[0], marker="*", label="best arms")
		if (d == 2):
			plt.plot(X[0,y1], X[1,y1], color=colours[1], marker="o", label="worst arms")
			plt.plot(X[0,y2], X[1,y2], color=colours[0], marker="o", label="best arms")
			for i in range(chunk_nb):
				ids = y == i+1
				idxs = np.argwhere(ids).flatten().tolist()
				slopes = np.array([X[1,x]/float(X[0,x]) if (X[0,x] != 0) else np.inf for x in idxs])
				intercepts = np.zeros(np.sum(ids))
				x_ = flatten_to_list(X[0,ids])
				y_ = flatten_to_list(X[1,ids])
				abline(x_, y_, slopes, intercepts, color=colours[i])
				ids[[y1,y2]] = False
				if (sum(ids.tolist()) > 0):
					plt.plot(X[0,ids], X[1,ids], color=colours[i], marker="o")
		if (d == 3):
			ax = plt.axes(projection='3d')
			for i in range(chunk_nb):
				ax.plot3D(X[0,y == i+1], X[1,y == i+1], X[2,y == i+1], c=X[2,y == i+1], cmap=colours[i])
		plt.legend(bbox_to_anchor=(0, 1), loc='best', ncol=1)
	else:
		#source: adapted from https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
		palette = ([(0, 1, 0.5, 1), (1, 0, 0, 1)] if (y[0] == 1) else [(1, 0, 0, 1), (0, 1, 0.5, 1)])
		feat_cols = ['Drug'+str(i) for i in range(X.shape[0])]
		df = pd.DataFrame(X.T,columns=feat_cols)
		df['group'] = y
		df['group'] = df['group'].apply(lambda i: "best arms" if (i==1) else "worst arms")
		df['Score class'] = df['group'].apply(lambda i: i)
		if (style == "PCA"):
			pca = PCA(n_components=ndim)
			pca_result = pca.fit_transform(df[feat_cols].values)
			df['pca-one'] = pca_result[:,0]
			if (ndim > 1):
				df['pca-two'] = pca_result[:,1]
				if (ndim > 2):
					df['pca-three'] = pca_result[:,2]
			print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
			if (ndim == 2):
				plt.figure(figsize=(16,10))
				sns.scatterplot(
					x="pca-one", y="pca-two",
					hue="group",
					palette=palette,
					data=df,
					legend="full",
					alpha=0.3
				)
			if (ndim == 3):
				ax = plt.figure(figsize=(16,10)).gca(projection='3d')
				ax.scatter(
					xs=df["pca-one"], 
					ys=df["pca-two"], 
					zs=df["pca-three"], 
					c=df["group"], 
					cmap='tab10'
				)
				ax.set_xlabel('pca-one')
				ax.set_ylabel('pca-two')
				ax.set_zlabel('pca-three')
		if (style == "TSNE"):
			if (d > 1000):
				pca_50 = PCA(n_components=50)
				df_values = pca_50.fit_transform(df[feat_cols])
				print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
			else:
				df_values = df[feat_cols]
			time_start = time.time()
			tsne = TSNE(n_components=ndim, verbose=1, perplexity=50, n_iter=300)
			tsne_results = tsne.fit_transform(df_values)
			print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
			df['tsne-one'] = tsne_results[:,0]
			if (ndim > 1):
				df['tsne-two'] = tsne_results[:,1]
				plt.figure(figsize=(16,10))
				sns.scatterplot(
				    x="tsne-one", y="tsne-two",
				    hue="group",
				    palette=palette,
				    data=df,
				    legend="full",
				    alpha=0.3
				)
	plt.savefig(path+"bandit_instance.png")

####################################################
## Saving & formatting results                    ##
####################################################

#' @param instance_path folder Python character string: folder must exist
#' @param data_name Python character string
#' @param K Python integer > 0
#' @param N Python integer > 0
#' @param m Python integer > 0 and < K
#' @param problem_type Python character string
#' @param omega_str Python character string
#' @return None (generates boxplot figure from previously executed runs)
def build_boxplot_per_experiment(instance_path, data_name, K, N, m, problem_type, omega_str=None, verbose=True):
	'''Generates boxplot to compare sample complexity of all tested algorithms on a single instance (@data_name, @K, @N, @m, @problem_type) in @instance_path folder (relative path from Code/)'''
	assert is_of_type(instance_path, "str")
	assert os.path.exists(instance_path)
	assert is_of_type(data_name, "str")
	assert is_of_type(K, "int")
	assert K > 0
	assert is_of_type(N, "int")
	assert N > 0
	assert is_of_type(m, "int")
	assert m > 0 and m < K
	assert is_of_type(problem_type, "str")
	assert is_of_type_OPTION(omega_str, "str")
	fsize = 24
	from glob import glob
	import subprocess as sb
	title = "---- Boxplot creation ----"
	print(("-"*len(title))+"\n"+title+"\n"+("-"*len(title)))
	fname = data_name+"_K"+str(K)+"_N"+str(N)+"_m"+str(m)+"_"+problem_type+("_"+str(omega_str) if (data_name == "classic") else "")
	bandit_instance = data_name+" ($K="+str(K)+"$, $N="+str(N)+"$, $m="+str(m)+"$)"
	bandit_instance = reduce(lambda x,y : x+"\_"+y, bandit_instance.split("_"))
	title = "Comparison of sample complexity for instance "+bandit_instance
	if (not os.path.exists(instance_path+"scores_"+problem_type+"_"+data_name+"_K="+str(K)+"_N="+str(N)+".csv")):
		print("Please run at least one algorithm on this instance!")
		raise ValueError
	with open(instance_path+"problem.txt", "r") as f:
		H = f.read().split("\n")[1].split(",")[1]
        plt.rcParams.update({'font.size': fsize})
	fig, ax = plt.subplots(figsize=(25,10), nrows=1, ncols=1)
	from bandits import bandit_factory
	algorithms = bandit_factory()
        import seaborn as sns
        algorithms_colours = dict(zip(algorithms, sns.color_palette(palette=None, n_colors=len(algorithms))))
	ci_algos = list(filter(lambda x : "Racing" in x or "LUCB" in x or "Gap" in x or "TS" in x or "GIFA" in x or "T3C" in x or "LinGame" in x, algorithms))
	classical_ci_algos = list(filter(lambda x : "Racing" in x or "KL-Racing" in x or ("LUCB" in x and not "LinLUCB" in x) or "KL-LUCB" in x or ("UGap" in x and not "LinUGapE" in x) or ("TS" in x and not "L-" in x) or ("T3C" in x and not "L-" in x), algorithms))
	algo_comp = []
	for algorithm in list(filter(lambda a : not("TrueUniform" in a), algorithms)):
		ls = glob(instance_path+"L-"+algorithm+"_*.txt")+glob(instance_path+algorithm+"_*.txt")
		if (len(ls) == 0):
			continue
                ## for closeup
                #if (not ("LinGapE" in algorithm) and not ("LinGIFA" in algorithm) and not ("LinGame" in algorithm)):
                #    continue
		print("---- Output of "+algorithm)
		ls = list(filter(lambda x : not("_hr" in x) and not("_recommendation" in x), ls))
		for l in ls:
			if ("L-"+algorithm in l):
				alg = "L-"+algorithm
			else:
				alg = algorithm
			## In order to make it easier to interpret
			if (m > 1 and alg == "LinGapE"):
				alg = "m-LinGapE"
			if (alg == "LinGIFAPlus"):
				alg = "LinGIFA (tau^LUCB)"
			if (alg == "LinIAA"):
				alg = "LinGIFA (individual indices)"
			mrn = l.split("/")[-1].split(".txt")[0].split("_")
			#comment
			beta = ""#mrn[1] if (algorithm in ci_algos) else ""
			greedy = ("largest variance" if (alg in classical_ci_algos) else "greedy") if (len("_".join(mrn).split("greedy")) > 1) else ""
			if (len(greedy) == 0 and alg == "LinGIFA"):
				greedy = "largest variance"
			calc = (beta if (len(beta) > 0) else "")+" "
			calc += (greedy if (len(greedy) > 0) else "")+(mrn[2] if (len(mrn) > 2 and len(greedy) == 0) else (" "+mrn[3] if (len(mrn) > 3) else ""))
			exploration = ("\n("+(calc if (len(calc) > 0) else "")+")") if (len(greedy)+len(beta)+(len(mrn)-2) > 0) else ""
			mat = np.loadtxt(l, skiprows=1, delimiter=",")
			if (np.shape(mat)[0] != 0):
				sample_complexity = np.asarray(mat[:,1], dtype=float).tolist()
				sample_complexity = [(iS+1)*S-iS*sample_complexity[iS-1] for iS, S in enumerate(sample_complexity)]
				assert all([s > 0 for s in sample_complexity])
                                alg_ = "LinGIFA" if ("LinGIFA (" in alg) else ("LinGapE" if (alg == "m-LinGapE") else alg)
				algo_comp.append([alg, exploration, sample_complexity, mat[-1,3], algorithms_colours[alg_]])
	sorted_algo_comp = list(sorted(algo_comp, key=lambda x : float(np.mean(x[2]))))
	medianprops = dict(linestyle='-', linewidth=2.5, color='lightcoral')
	meanpointprops = dict(marker='.', markerfacecolor='white', alpha=0.)
	labels = [ls[0]+" "+ls[1]+"\nerror="+str(round((100.-ls[3])/100., 5) if (round((100.-ls[3])/100., 5) > 0) else 0) for ls in sorted_algo_comp]
	from seaborn import boxplot, stripplot
	meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='black', markersize=10)
	bplot = sns.boxplot(data=[ls[2] for ls in sorted_algo_comp], ax=ax, showmeans=False)#True, meanprops=meanprops)
        for i in range(len(sorted_algo_comp)):
            mybox = bplot.artists[i]
            mybox.set_facecolor(sorted_algo_comp[i][-1])
        bplot = sns.stripplot(data=[ls[2] for ls in sorted_algo_comp], jitter=True, marker='o', alpha=0.25, color="grey")
        ax.plot(ax.get_xticks(), [np.mean(ls[2]) for ls in sorted_algo_comp], "kD", label="means", markersize=20)
	#cm = plt.get_cmap('gist_rainbow')
	#N = len(sorted_algo_comp)
	#colors = [cm(1.*i/N) for i in range(N)]
	ax.set_yticklabels([int(ytick) for ytick in ax.get_yticks()], fontsize=fsize)
	ax.set_xticklabels(labels, rotation=27, fontsize=fsize)
	ax.set_ylabel("Sample complexity", fontsize=fsize)
        plt.legend()
	#ax.set_xlabel("Algorithm", fontsize=fsize)
	#ax.set_title(title, fontsize=fsize)
	boxplot_file = instance_path+fname+".png"
	plt.savefig(boxplot_file, bbox_inches="tight")
	print("Boxplot successfully saved in '"+boxplot_file+"'")

#' @param instance_path folder Python character string: folder must exist
#' @param data_name Python character string
#' @param K Python integer > 0
#' @param N Python integer > 0
#' @param m Python integer > 0 and < K
#' @param problem_type Python character string
#' @param omega_str Python character string
#' @return None (generates LaTeX file from previously executed runs)
def generate_latex_file(instance_path, data_name, K, N, m, problem_type, omega_str=None, verbose=True, latex_compiler="pdflatex"):
	'''Generates LaTex source file for the considered bandit instance (@data_name, @K, @N, @m, @problem_type) in @instance_path folder (relative path from Code/)'''
	assert is_of_type(instance_path, "str")
	assert os.path.exists(instance_path)
	assert is_of_type(data_name, "str")
	assert is_of_type(K, "int")
	assert K > 0
	assert is_of_type(N, "int")
	assert N > 0
	assert is_of_type(m, "int")
	assert m > 0 and m < K
	assert is_of_type(problem_type, "str")
	assert is_of_type_OPTION(omega_str, "str")
	from glob import glob
	import subprocess as sb
	title = "---- LateX file creation ----"
	print(("-"*len(title))+"\n"+title+"\n"+("-"*len(title)))
	fname = data_name+"_K"+str(K)+"_N"+str(N)+"_m"+str(m)+"_"+problem_type+("_"+str(omega_str) if (data_name == "classic") else "")
	bandit_instance = data_name+" ($K="+str(K)+"$, $N="+str(N)+"$, $m="+str(m)+"$)"
	bandit_instance = reduce(lambda x,y : x+"\_"+y, bandit_instance.split("_"))
	header="\documentclass{report}\n\usepackage[utf8]{inputenc}\n\usepackage[english]{babel}\n\usepackage{amssymb}\n\usepackage{amsmath}\n\usepackage{graphicx}\n\usepackage{geometry}[margins=8in]\n\/title{Bandit Instance: "+bandit_instance+"}\n\/begin{document}\n\maketitle"
	if (verbose):
		print("- Header created")
	footer="\end{document}"
	if (verbose):
		print("- Footer created")
	if (not os.path.exists(instance_path+"scores_"+problem_type+"_"+data_name+"_K="+str(K)+"_N="+str(N)+".csv")):
		raise ValueError("Please run at least one algorithm on this instance!")
	if (K <= 50):
		scores_table="\/begin{table}[H]\n\centering\n\/begin{tabular}{c|c}\n\hline\n\/textbf{Arm index $i$} & \/textbf{Score = true average reward $\mu_i$}\/\/\n"
		array = np.loadtxt(instance_path+"scores_"+problem_type+"_"+data_name+"_K="+str(K)+"_N="+str(N)+".csv").tolist()
		sorted_scores = np.argsort(array).tolist()
		for t in sorted_scores:
			s = array[t]
			scores_table += "$"+str(t)+"$ & $"+str(round(s, 3))+"$ \/\/\n"
		scores_table += "\hline\n\end{tabular}\caption{Scores in "+bandit_instance+"}\n\end{table}\n"
	else:
		scores_table = ""
	if (verbose):
		print("- Score table created")
	str_ = ""
	parameters_table="\/begin{table}[H]\n\centering\n\/begin{tabular}{c|c}\n\hline\n\/textbf{Parameter} & \/textbf{Value}\/\/\n\hline\n"
	with open(instance_path+"parameters.json", "r") as f:
		str_ = f.read()
		str_ = str_[1:-1]
		str_ls = [x.split(": ") for x in str_.split(", ")]
		for x in str_ls:
			x[0] = x[0][1:-1]
			if (x[0] in ["alpha", "delta", "eta", "k1_diff", "n_simu", "prob", "sigma"]):
				if (x[0] == "n_simu"):
					nsimu = int(x[1])
					x[0] = "n\_simu"
				if (x[0] == "k1_diff"):
					x[0] = "k1\_diff"
				if (x[0] == "delta"):
					delta = float(x[1])
				try:
					float(x[1])
					value = "$"+x[1]+"$"
				except:
					value = x[1]
				if (x[0] in ["alpha", "delta", "eta", "sigma"]):
					parameters_table += "$\\"+x[0]+"$ & "+value+" \/\/\n"
				else:
					parameters_table += x[0]+" & "+value+" \/\/\n"
	parameters_table += "\hline\n\end{tabular}\n\caption{Parameter values in "+bandit_instance+"}\n\end{table}\n"
	if (verbose):
		print("- Parameter table created")
	problem_table = "\/begin{table}[H]\n\centering\n\/begin{tabular}{c|c}\n\hline\n\/textbf{Measures} & \/textbf{Values}\/\/\n\hline\n"
	with open(instance_path+"problem.txt", "r") as f:
		str_ = f.read().split("\n")[1].split(",")
	problem_table += "\/textbf{Gap ($\mu_m-\mu_{m+1}$)} & $"+str_[0]+"$\/\/\n"
	problem_table += "\/textbf{Expected number of samples} (as computed in the UGapE paper) & $\mathcal{O}("+str_[1]+")$\/\/\n"
	problem_table += "\hline\n\end{tabular}\n\end{table}\n"	
	if (verbose):
		print("- Problem table created")
	figures_plot = "\/begin{figure}[H]\n\centering\n\includegraphics[scale=0.3]{"+instance_path+"bandit_instance.png}\n\caption{Visualization of Feature Vectors in "+bandit_instance+"}\n\end{figure}\n"
	figures_plot += "\/begin{figure}[H]\n\centering\n\includegraphics[scale=0.4]{"+instance_path+"problem_scores.png}\n\caption{Histogram of Scores in "+bandit_instance+"}\n\end{figure}\n"
	if (verbose):
		print("- Figure plot created")
	if (os.path.exists(instance_path+"/T_init_finetuning.png")):
		figures = ["\/begin{figure}[H]\n\centering\n\includegraphics[scale=0.44]{"+instance_path+"/T_init_finetuning.png}\n\caption{}\n\end{figure}\n"]
	else:
		figures = []
	from bandits import bandit_factory
	algorithms = bandit_factory()
	ci_algos = list(filter(lambda x : "Racing" in x or "LUCB" in x or "Gap" in x or "TS" in x or "GIFA" in x or "T3C" in x or "LinGame" in x, algorithms))
	for algorithm in algorithms:
		ls = glob(instance_path+"L-"+algorithm+"_*.png")+glob(instance_path+algorithm+"_*.png")
		if (len(ls) == 0):
			continue
		print("---- Plot of "+algorithm)
		for l in ls:
			if ("L-"+algorithm in l):
				alg = "L-"+algorithm
			else:
				alg = algorithm
			mrn = l.split("/")[-1].split(".txt")[0].split("_")
			beta = mrn[1].split(".")[0] if (algorithm in ci_algos) else ""
			greedy = mrn[-1].split(".")[0] if (len(mrn) > 2) else ""
			calc = (beta if (len(beta) > 0) else "")+" "
			calc += (greedy if (len(greedy) > 0) else "")+(mrn[2] if (len(mrn) > 2 and len(greedy) == 0) else (" "+mrn[3] if (len(mrn) > 3) else ""))
			exploration = ("("+(beta if (len(beta) > 0) else "")+(" - "+calc if (len(calc) > 0) else "")+")") if (len(greedy)+len(beta)+(len(mrn)-2) > 0) else ""
			figures += ["\/begin{figure}[H]\n\centering\n\includegraphics[scale=0.44]{"+l+"}\n\caption{"+alg+" in "+bandit_instance+" "+exploration+"}\n\end{figure}\n"]
	if (verbose):
		print("- Plots of results created")
	comparative_table = "\/begin{table}[H]\n\centering\n\/begin{tabular}{|l|c|c|c|}\n\hline\n\/textbf{Algorithm} & \/textbf{Empirical Sample} & \/textbf{Empirical Standard} & \/textbf{Error Frequency}\/\/\n& \/textbf{Complexity} & \/textbf{Deviation} & \/\/\n\hline\n"
	ls = glob(instance_path+"TrueUniform_finetuning_T_init.txt")
	if (len(ls) > 0):
		mat = np.loadtxt(ls[0], skiprows=1, delimiter=",")
		if (np.shape(mat)[0] != 0):
			ids = flatten_to_list(np.argwhere(mat[:,3] >= (1-delta)*100))
			if (len(ids) != 0):
				comparative_table += "TrueUniform ($T\_init="+str(int(mat[ids[0],1]))+"$) & $"+str(int(mat[ids[0],2]))+"$ &  & $"+str(round(100-mat[ids[0],3], 3))+"\%$\/\/\n"
			idx = int(np.argmax(mat[:,3]))
			comparative_table += "TrueUniform ($T\_init="+str(int(mat[idx,1]))+"$) & $"+str(int(mat[idx,2]))+"$ &  & $"+str(round(100-mat[idx,3], 3))+"\%$\/\/\n"
			comparative_table += "\hline\n"
	algo_comp = []
	for algorithm in list(filter(lambda a : not("TrueUniform" in a), algorithms)):
		ls = glob(instance_path+"L-"+algorithm+"_*.txt")+glob(instance_path+algorithm+"_*.txt")
		if (len(ls) == 0):
			continue
		print("---- Output of "+algorithm)
		ls = list(filter(lambda x : not("_hr" in x) and not("_recommendation" in x), ls))
		for l in ls:
			if ("L-"+algorithm in l):
				alg = "L-"+algorithm
			else:
				alg = algorithm
			mrn = l.split("/")[-1].split(".txt")[0].split("_")
			beta = mrn[1] if (algorithm in ci_algos) else ""
			greedy = "greedy" if (len("_".join(mrn).split("greedy")) > 1) else ""
			calc = (greedy if (len(greedy) > 0) else "")+(mrn[2] if (len(mrn) > 2 and len(greedy) == 0) else (" "+mrn[3] if (len(mrn) > 3) else ""))
			exploration = ("("+(beta if (len(beta) > 0) else "")+(" - "+calc if (len(calc) > 0) else "")+")") if (len(greedy)+len(beta)+(len(mrn)-2) > 0) else ""
			mat = np.loadtxt(l, skiprows=1, delimiter=",")
			if (np.shape(mat)[0] != 0):
				algo_comp.append([alg, exploration, mat[-1,1], mat[-1,2], mat[-1,3]])
	sorted_algo_comp = list(sorted(algo_comp, key=lambda x : x[2]))
	for x in sorted_algo_comp:
		comparative_table += x[0]+" "+x[1]+" & $"+str(int(x[2]))+"$ & $"+str(int(x[3]))+"$ & $"+str(round(100-x[4], 3))+"\%$\/\/\n"
	comparative_table += "\hline\n\end{tabular}\n\caption{Comparative Table for "+bandit_instance+" ($n\_simu = "+str(nsimu)+"$).}\n\end{table}\n\n"
	if (verbose):
		print("- Comparative table created")
	latex_file = instance_path+fname+".tex"
	with open(latex_file, "w+") as f:
		latex = [header, scores_table, parameters_table, problem_table]
		latex += [comparative_table]
		f.write(reduce(lambda x,y : x+"\n"+y, latex))
	sb.call("sed -i 's/[/]/[]/g' "+latex_file, shell=True)
	sb.call("sed -i 's/[[][]]//g' "+latex_file, shell=True)
	with open(latex_file, "a+") as f:
		latex = [figures_plot]+figures+[footer]
		f.write(reduce(lambda x,y : x+"\n"+y, latex))
	sb.call("sed -i 's/[/]begin/[]begin/g' "+latex_file, shell=True)
	sb.call("sed -i 's/[[][]]//g' "+latex_file, shell=True)
	sb.call("sed -i 's/[[]H[]]//g' "+latex_file, shell=True)
	sb.call(latex_compiler+" "+latex_file, shell=True)
	sb.call("rm -f *.log *.aux", shell=True)
	sb.call("mv "+fname+".pdf "+instance_path, shell=True)
	print("LaTeX file has been successfully created and stored in '"+latex_file+"'")

####################################################
## Scoring function                               ##
####################################################

#' @param x NumPy array of float of size n
#' @param P NumPy array of float of size n
#' @return Python float: cosine score value
def cosine_score(x, P, type_=["similarity", "dissimilarity"][0], scale=False):
	'''Computes cosine score S : x, P -> 1-(x.P)/(||x||.||P||)'''
	assert type_ in ["similarity", "dissimilarity"]
	#assert is_of_type_LIST([x, P], "numpy.ndarray") and x.size == P.size
	df = x.join(P, how="inner")
	if (scale):
		scale_func = lambda col : (np.array(col)-np.mean(col))/np.std(col)
		df = df.apply(lambda col : scale_func(col.values.flatten().tolist()))
	x, P = df[[df.columns[0]]], df[[df.columns[1]]]
	cos = float(np.dot(np.transpose(x), P)/(np.linalg.norm(x, 2)*np.linalg.norm(P, 2)))
	if (type_ == "dissimilarity"):
		return 1.-cos
	return cos

from sklearn.metrics import jaccard_similarity_score
def tanimoto(x, P):
	'''Computes Jaccard score S : x, P -> 1-(x & P)/(x | P)'''
	#assert is_of_type_LIST([x, P], "numpy.ndarray") and x.size == P.size
	df = x.join(P, how="inner")
	##print(len(df.index)) 11 per sample -> 50 phenotype
	x, P = df[[df.columns[0]]].T.values.flatten(), df[[df.columns[1]]].T.values.flatten()
	return jaccard_similarity_score(x,P)

###############################################################################
## Kullback-Leibler divergences for 1D exponential families                  ##
###############################################################################

def rel_entropy(x,y):
	if (x==0):
		return 0.
	if (y==0):
		return 1.
	return x*np.log(x/float(y))

klGauss = lambda x,y,sigma2 : (x-y)**2/float(2*sigma2)
klBern = lambda x,y : rel_entropy(x,y)+rel_entropy(1-x,1-y)
klExp = lambda x,y : float("inf") if (y==0) else (x/float(y)-np.log(x/float(y))-1)
klPoisson = lambda x,y : rel_entropy(x,y)-x+y

## solution for y in min_y { d(x, y) - y x }
invGauss = lambda x,y,sigma2 : x+y*sigma2
invBern = lambda x,y : 2*x/float(1-y+np.sqrt((y-1)**2+4*x*y))
invExp = lambda x,y : float("inf") if (y>0) else (2*x/float(1+np.sqrt(1-4*x*y)))
invPoisson = lambda x,y : x/float(1-y)

## binary search
def klucb(x, level, div, lowerbound=-float('inf'), upperbound=float('inf'), precision=1e-10, maxiter=100):
	"""Generic klUCB index computation using binary search: 
	returns u such that div(x,u)=level where div is the KL divergence to be used.
	"""
	if (div(x, lowerbound) > level):
		return lowerbound
	if (div(x, upperbound) < level):
		return upperbound
	for i in range(maxiter):
		m = (lowerbound+upperbound)/2.
		if (m in [lowerbound, upperbound]):
			return m
		fm = div(x, m)
		if (fm < -precision):
			lowerbound = m
		elif (fm > precision):
			upperbound = m
		else:
			return m
	return (lowerbound+upperbound)/2.

def klucbGauss(x, level, sig2=1., lower=float("-inf"), upper=float("inf")):
	"""returns u such that kl(x,u)=level for the Gaussian kl-divergence (can be done in closed form)."""
	if (upper==x):
		return x - np.sqrt(2*sig2*level)
	elif (lower==x):
		return x + np.sqrt(2*sig2*level)
	else:
		raise ValueError

def klucbBern(x, level, lower=float("-inf"), upper=float("inf")):
	"""returns u such that kl(x,u)=level for the Bernoulli kl-divergence."""
	if (lower==x):
		if (lower==1):
			return 1.
		return klucb(x, 0., lambda a,b : klBern(a, b)-level, lowerbound=x, upperbound=1.)
	elif (upper==x):
		if (upper==0):
			return 0.
		return klucb(x, 0., lambda a,b : level-klBern(a, b), lowerbound=0., upperbound=x)
	else:
		raise ValueError

def klucbPoisson(x, level, lower=float("-inf"), upper=float("inf")):
	"""returns u such that kl(x,u)=level for the Poisson kl-divergence."""
	if (lower==x):
		klucb(x, level, klPoisson(a,b), lowerbound=x, upperbound=x+level+np.sqrt(level*level+2*x*level))
	elif (upper==x):
		klucb(x, level, klPoisson(a,b), lowerbound=lower, upperbound=x)
	else:
		raise ValueError

def klucbExp(x, level, lower=float("-inf"), upper=float("inf")):
	"""returns u such that kl(x,u)=d for the exponential kl divergence."""
	if (lower==x):
		return klucb(x, 0., lambda a,b : klExp(a,b)-level, lowerbound=x, upperbound=x*np.exp(level+1))
	elif (upper==x):
		return klucb(x, 0., lambda a,b : level-klExp(a,b), lowerbound=0., upperbound=x)
	else:
		raise ValueError

#_____________________________________________________________________________________________

kl_di_bounds = lambda sigma : {
	"gaussian": lambda x,y,z,a : klucbGauss(x, y, sig2=sigma**2, lower=z, upper=a), 
	"linlucb": lambda x,y,z,a : klucbGauss(x, y, sig2=sigma**2, lower=z, upper=a), 
	"bernouilli": lambda x,y,z,a : klucbBern(x, y, lower=z, upper=a), 
	"epilepsy": lambda x,y,z,a : klucbBern(x, y, lower=z, upper=a), 
	"poisson": lambda x,y,z,a : klucbPoisson(x, y, lower=z, upper=a), 
	"exponential": lambda x,y,z,a : klucbExp(x, y, lower=z, upper=a)
}

prior_di = {
	"bernouilli": np.random.beta,
	"epilepsy": np.random.beta,
	"gaussian": np.random.normal,
	"linlucb": np.random.normal,   
	"poisson": np.random.gamma, 
	"exponential": np.random.gamma,
}

prior_args_di = lambda eta, K : {
	"bernouilli": np.matrix([[1]*K, [1]*K]),
	"epilepsy": np.matrix([[1]*K, [1]*K]),
	"gaussian": np.matrix([[0]*K, [eta]*K]),
	"linlucb": np.matrix([[0]*K, [eta]*K]),
	"poisson": np.matrix([[0]*K, [eta]*K]),
	"exponential": np.matrix([[0]*K, [eta]*K])
}

prior_pdfs = {
	"bernouilli": lambda x, args : beta.pdf(x, args[0], args[1]),
	"epilepsy": lambda x, args : beta.pdf(x, args[0], args[1]),
	"gaussian": lambda x, args : norm.pdf(x, loc=args[0], scale=args[1]),
	"linlucb": lambda x, args : norm.pdf(x, loc=args[0], scale=args[1]),
	"poisson": lambda x, args : gamma.pdf(x, a=args[0], scale=args[1]), 
	"exponential": lambda x, args : gamma.pdf(x, a=args[0], scale=args[1]),
}

prior_cdfs = {
	"bernouilli": lambda x, args : beta.cdf(x, args[0], args[1]),
	"epilepsy": lambda x, args : beta.cdf(x, args[0], args[1]),
	"gaussian": lambda x, args : norm.cdf(x, loc=args[0], scale=args[1]),
	"linlucb": lambda x, args : norm.cdf(x, loc=args[0], scale=args[1]),  
	"poisson": lambda x, args : gamma.cdf(x, a=args[0], scale=args[1]), 
	"exponential": lambda x, args : gamma.cdf(x, a=args[0], scale=args[1]),
}

kl_di = lambda sigma : {
	"gaussian": lambda x,y : klGauss(x, y, sigma**2), 
	"epilepsy": lambda x,y : klBern(x, y),  
	"bernouilli": lambda x,y : klBern(x, y), 
	"poisson": lambda x,y : klPoisson(x, y), 
	"exponential": lambda x,y : klExp(x, y)
}

###############################################################################
## Kullback-Leibler divergences for exponential families                     ##
###############################################################################

from random import sample

#' @param x Python list of float or int
#' @param f Python function float Numpy array -> float or int Numpy array -> int
#' @return smp return an element e in x which satisfies
def randf(x, f):
	x_float = is_of_type_LIST(x, "float")
	assert x_float or is_of_type_LIST(x, "int")
	t = (float if (is_of_type_LIST(x, "float")) else int)
	x = np.array(x, dtype=t)
	f_ = lambda z : t(f(z))
	assert is_of_type(f_(x), "float" if (x_float) else "int")
	smp = sample(np.argwhere(x == f_(x)).flatten().tolist(), 1)[0]
	assert is_of_type(smp, "int")
	return smp

#' @param x Python list of float or int of length K
#' @param m Python integer > 0 and <= K
#' @return res Python list of length @m of maximizing elements in @x IN ORDER (ties broken randomly)
def m_maximal(x, m):
	'''Returns @m-sized list of maximizing elements in @x IN ORDER (ties broken randomly)'''
	assert is_of_type_LIST(x, "float") or is_of_type_LIST(x, "int")
	assert is_of_type(m, "int")
	K = len(x)
	assert m > 0 and m <= K
	ids = np.argsort(x).flatten().tolist()
	ids.reverse()
	if (m < K):
		ids = [None]*m
		for i in range(m):
			id_ = randf(x, np.max)
			ids[i] = id_
			x[id_] = -float("inf")
	return ids

#' @param values Python list of float or int of length K
#' @param m Python integer > 0 and <= K
#' @return res Python element in values which is the mth maximal element (ties broken randomly)
def m_max(values, m):
	'''Returns element in @values which is the @mth maximal element'''
	assert is_of_type_LIST(values, "float") or is_of_type_LIST(values, "int")
	assert is_of_type(m, "int")
	arg_values = np.argsort(np.array(values).flatten()).tolist()
	assert is_of_type(arg_values[-m], "int")
	assert is_of_type(values[arg_values[-m]], "float" if (is_of_type_LIST(values, "float")) else "int")
	return values[arg_values[-m]]

#' @param x NumPy NDarray of size N x 1
#' @param A NumPy matrix of size  N x N
#' @return ||x||_{A} (matrix norm)
def matrix_norm(x, A):
	'''Mahalanobis norm/Matrix norm: sqrt(x^TAx)'''
	assert (is_of_type(x, "numpy.ndarray") or is_of_type(x, "numpy.matrix")) and is_of_type(A, "numpy.matrix") or is_of_type(A, "numpy.ndarray")
	assert len(np.shape(x)) == 2 and len(np.shape(A)) == 2
	assert np.shape(x)[0] == np.shape(A)[0] and np.shape(A)[0] == np.shape(A)[1]
	return np.sqrt((x.T).dot(A).dot(x))
