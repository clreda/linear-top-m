# coding: utf-8

import csv
import numpy as np
import pandas as pd

from utils import is_of_type, is_of_type_LIST, is_of_type_OPTION, binarize_via_histogram, apply_mask, get_state, write_observations_file, get_drug_id, cosine_score, tanimoto
from constants import dr_folder, target_folder, folder_path

####################################################
## (Arm) problems                                 ##
####################################################

#' @param scores Python float list
#' @param args Python dictionary
class GenericProblem(object):
	'''Defines a problem with available "intrinsic" scores. Call to black-box function is simulated by I/O operations'''
	def __init__(self, scores, dataname, problem_type="generic", args=None, path_to_plots=folder_path):
		assert is_of_type_LIST(scores, "float")
		assert is_of_type_OPTION(args, "dict")
		assert "X" in list(args.keys())
		N, K = np.shape(args["X"])
		self.scores_filepath = path_to_plots+"scores_"+problem_type+"_"+dataname+"_K="+str(K)+"_N="+str(N)+".csv"
		self.X_filepath = path_to_plots+"features_"+problem_type+"_"+dataname+"_K="+str(K)+"_N="+str(N)+".csv"
		self.args = args
		self.oracle = scores
		with open(self.scores_filepath, "w") as writeFile:
			writer = csv.writer(writeFile)
			writer.writerows([[s] for s in self.oracle])
		np.savetxt(self.X_filepath, args["X"])
		self.type = problem_type

	def get_reward(self, arm):
		## File IO error if several threads use the same file
		if (False):
			with open(self.scores_filepath, "r") as readFile:
				reader = csv.reader(readFile, delimiter=',')
				for row, record in enumerate(reader):
					if (row == arm):
						break
			try:
				record
			except:
				print(self.scores_filepath)
				print(os.path.exists(self.scores_filepath))
				print("Resource is already used by another thread")
				raise ValueError
		else:
			record = [self.oracle[arm]]
		return self.reward(float(record[0]))	

	def reward(self, score):
		raise NotImplemented

## Gaussian arms
class GaussianProblem(GenericProblem):
	'''Subclass of GenericProblem. Defines a Generic Problem with Gaussian rewards'''
	def __init__(self, scores, dataname, args, path_to_plots=folder_path):
		assert "sigma" in list(args.keys())
		super(GaussianProblem, self).__init__(scores, dataname, problem_type="gaussian", args=args, path_to_plots=path_to_plots)
		self.sigma = self.args["sigma"]

	def reward(self, score):
		return np.random.normal(score, self.sigma)

## Bernouilli arms
class BernouilliProblem(GenericProblem):
	'''Subclass of GenericProblem. Defines a Generic Problem with Bernouilli rewards'''
	def __init__(self, scores, dataname, args, path_to_plots=folder_path):
		assert all(list(map(lambda x : x <= 1 and x >= 0, scores)))
		super(BernouilliProblem, self).__init__(scores, dataname, problem_type="bernouilli", args=args, path_to_plots=path_to_plots)

	def reward(self, score):
		return np.random.binomial(1, score)

## Poisson arms
class PoissonProblem(GenericProblem):
	'''Subclass of GenericProblem. Defines a Generic Problem with Poisson rewards'''
	def __init__(self, scores, dataname, args, path_to_plots=folder_path):
		if (any([x < 0 for x in scores])):
			scores -= np.min(scores)
		assert all(list(map(lambda x : x >= 0, scores)))
		super(PoissonProblem, self).__init__(scores, dataname, problem_type="poisson", args=args, path_to_plots=path_to_plots)

	def reward(self, score):
		return np.random.poisson(score)

## Exponential arms
class ExponentialProblem(GenericProblem):
	'''Subclass of GenericProblem. Defines a Generic Problem with Exponential rewards'''
	def __init__(self, scores, dataname, args, min_score=1e-3, path_to_plots=folder_path):
		if (any([x <= 0 for x in scores])):
			scores -= np.min(scores)-min_score
			scores = list(map(float, scores))
		assert all(list(map(lambda x : x > 0, scores)))
		super(ExponentialProblem, self).__init__(scores, dataname, problem_type="exponential", args=args, path_to_plots=path_to_plots)

	def reward(self, score):
		return np.random.exponential(score)

## DR instance: arms with dynamically obtained scores using the GRN
class DRProblem(GenericProblem):
	'''Subclass of GenericProblem. Defines a Generic Problem with simulated rewards using scores from GRN'''
	def __init__(self, scores, dataname, args, func="cosine", path_to_plots=folder_path, quiet=True):
		assert all(list(map(lambda x : x in [-1, 0, 1], scores)))
		assert all([key in list(args.keys()) for key in ["S", "X", "names", "grn_name", "path_to_grn"]])
		import subprocess as sb
		import os
		from glob import glob
		self.scores = list(map(float, scores))
		self.names =  list(map(str, args["names"]))
		from constants import dr_folder
		self.grn_name = args["grn_name"]
		grn_name = self.grn_name.split("/")[-1]
		self.path_to_grn = args["path_to_grn"]
		## Store in the expansion-network program the GRN file
		if (not os.path.exists(self.path_to_grn+grn_name)):
			if (not os.path.exists(self.path_to_grn)):
				sb.call("mkdir "+self.path_to_grn, shell=True)
			sb.call("cp "+dr_folder+self.grn_name+" "+self.path_to_grn, shell=True)
		with open(dr_folder+self.grn_name, "r") as f:
			model = f.read()
			self.genes = [x.split("[")[0].upper() for x in model.split("\n")[5].split("; ")][:-1]
		## Note in the model that any gene in the GRN can be perturbed
		with open(self.path_to_grn+grn_name, "w") as f:
			model = model.split("\n")
			buildgene = lambda x : x.split("[")[0]+"[+-]{"+x.split("{")[-1].split("}")[0]+"}("+x.split("(")[-1].split(")")[0]+")"
			model[5] = "; ".join([buildgene(x) for x in model[5].split("; ")][:-1])+"; "
			f.write("\n".join(model))
		self.perts = ["-+"]*len(self.genes)
		## scoring function between final attractor state and "differential phenotype"
		di_scores = {"adj_cosine" : lambda A, B : cosine_score(A, B, scale=True, type_="similarity"), 
			"cosine": lambda A, B : cosine_score(A, B, scale=False, type_="similarity"), 
			"tanimoto": lambda A, B : tanimoto(A, B),
			"discosine": lambda A, B : cosine_score(A, B, type_="dissimilarity", scale=False)}
		assert func in list(di_scores.keys())
		self.func = di_scores[func]
		self.baseline = di_scores["discosine"]
		## drug signatures
		# non binarized (in order to compute the baseline method)
		self.X = args["X"]
		# binarized (what is used to generate the perturbation due to drug treatment)
		self.S = None if ("S" not in list(args.keys())) else args["S"]
		## differential phenotype for disease 
		phenotype = pd.read_csv(dr_folder+dataname+"_phenotype.csv")
		phenotype.index = phenotype[phenotype.columns[0]]
		phenotype = phenotype.drop(columns=[phenotype.columns[0]])
		phenotype = phenotype.loc[~phenotype.index.duplicated()]
		phenotype = phenotype.loc[list(filter(lambda x : x in self.genes, phenotype.index))]
		self.phenotype = pd.DataFrame([1-int(s) for s in phenotype[phenotype.columns[0]]], index=phenotype.index, columns=["-"+phenotype.columns[0]])
		## decides whether to compute the baseline method scores
		if (not quiet and os.path.exists(dr_folder+dataname+"_phenotype_nonbinarized.csv")):
			phenotype = pd.read_csv(dr_folder+dataname+"_phenotype_nonbinarized.csv")
			phenotype.index = phenotype[phenotype.columns[0]]
			phenotype = phenotype.drop(columns=[phenotype.columns[0]])
			phenotype = phenotype.loc[~phenotype.index.duplicated()]
			phenotype = phenotype.loc[list(filter(lambda x : x in self.genes, phenotype.index))]
			self.diff_treated_control = phenotype
		else:
			self.diff_treated_control = None
		## quantile normalized phenotype of controls
		controls = pd.read_csv(dr_folder+"GSE77578_controls.csv")
		controls.index = controls[controls.columns[0]]
		controls = controls.drop(columns=[controls.columns[0]])
		controls = controls.loc[~controls.index.duplicated()]
		self.controls = controls
		## quantile normalized phenotype of patients
		patients = pd.read_csv(dr_folder+"GSE77578_patients.csv")
		patients.index = patients[patients.columns[0]]
		patients = patients.drop(columns=[patients.columns[0]])
		patients = patients.loc[~patients.index.duplicated()]
		self.patients = patients
		super(DRProblem, self).__init__(self.scores, dataname, problem_type="epilepsy", args={"X": self.X.values}, path_to_plots=path_to_plots)

	def compute_score_from_grn(self, grn_sig, aggregate=["differential", "mean"][0]):
		from scipy.stats import pearsonr
		from utils import cosine_score, tanimoto
		## Another possibility is to return the mean of the scoring function across all control samples
		## instead of considering the "differential phenotype" computed above
		if (aggregate == "mean"):
			nsamples = np.shape(self.controls.values)[1]
			def binarize_samples(df, idx):
				df = binarize_via_histogram(df)
				df.columns = [idx]
				return df
			controls = reduce(lambda x,y : x.join(y, how="outer"), [binarize_samples(self.controls[[sample]], sample) for si, sample in enumerate(self.controls.columns)])
			df = grn_sig.join(controls, how="outer").dropna()
			allsamples = [df.columns[s+1] for s in range(nsamples)]
			scores = [self.func(df[[df.columns[0]]], df[[sample]]) for sample in allsamples]
			score = np.mean(scores)
		else:
			score = self.func(grn_sig, self.phenotype)
		return float(score)

	def print_state(self, state):
		ones = pd.DataFrame([1]*len(self.genes), index=self.genes, columns=["ones"])
		df = state.join(ones, how="outer").fillna(-1)
		df = df[[df.columns[0]]]
		df[df.columns[0]] = np.asarray(np.asarray(df[[df.columns[0]]].values, dtype=int), dtype=str)
		df[df == "-1"] = "|"
		print("".join(df[df.columns[0]].values.flatten().tolist())+"\t"+state.columns[0])

	def get_reward(self, arm, quiet=False, sample_id=None):
		import subprocess as sb
		if (not is_of_type(sample_id, "int")):
			from random import sample
			sample_id = sample(range(np.shape(self.patients.values)[1]), 1)[0]
		initial = binarize_via_histogram(self.patients[[self.patients.columns[sample_id]]])
		initial = initial.loc[list(filter(lambda x : x in initial.index, self.genes))].dropna()
		initial.columns = ["initial"]
		if (not quiet):
			self.print_state(initial)
		sig = self.S[[self.S.columns[arm]]]
		sig = sig.loc[list(filter(lambda x : x in sig.index, self.genes))].dropna()
		sig.columns = ["drug"]
		if (not quiet):
			self.print_state(sig)
		masked = apply_mask(initial, sig).dropna()
		if (not quiet):
			self.print_state(masked)
		observations_name = "observations_prediction"
		observations_fname = self.path_to_grn+"/"+observations_name+".spec"
		length, solmax = 40, 1
		experiments = [{"cell": "Cell", "dose": "NA", "pert": "arm "+str(arm), "ptype": "trt_cp", "itime": "NA", "perturbed_oe": [], "perturbed_ko": [], "exprs": {"Initial": {"step": 0, "sig": masked}}}]
		N = write_observations_file(observations_fname, length, self.perts, self.genes, experiments, verbose=False)
		assert N > 0
		ko_exists = any([v in self.perts for v in ["-", "+-", "-+"]])
		fe_exists = any([v in self.perts for v in ["+", "+-", "-+"]])
		cmd = "cd "+self.path_to_grn.split("examples/")[0]+"Python ; python solve.py launch "+self.path_to_grn.split("/")[-2]+" --model "+self.grn_name.split("/")[-1].split(".net")[0]
		cmd += " --experiments "+observations_name+" --q0 Initial1 --nstep "+str(length)+" --solmax "+str(solmax)
		cmd += (" --KO KnockDown1" if (ko_exists) else "")+(" --FE OverExpression1" if (fe_exists) else "")
		cmd += " --modelID 0 --steadyStates 1"
		output = sb.check_output(cmd, shell=True)
		try:
			treated, _ = get_state(output, length, self.genes, solmax=solmax)
			if (not quiet):
				self.print_state(treated)
				self.print_state(self.phenotype)
			score = self.compute_score_from_grn(treated)
		except:
			score = -666
		if (not quiet and str(self.diff_treated_control) != "None"):
			from utils import cosine_score
			sig = self.X[[self.X.columns[arm]]]
			sig = sig.loc[list(filter(lambda x : x in sig.index, self.genes))].dropna()
			sig.columns = ["drug"]
			dcosine = self.baseline(sig, self.diff_treated_control)
			print(("Arm: ", (arm, self.names[arm]), "predicted, true scores, signature = ", (score, self.scores[arm], dcosine)))
		elif (not quiet):
			print(("Arm: ", (arm, self.names[arm]), "predicted, true scores = ", (score, self.scores[arm])))
		return score

## "easy" DR instance: in order to save some time, using arms with statically obtained scores using the GRN
## rewards from every sample, every drug has been saved
## Using the 10 arms among the whole set of drugs
class DRProblemSubset(DRProblem):
	'''Subclass of DRProblem on subset of 10 drugs (5+, 5-) and 18 patient samples.'''
	def __init__(self, scores, dataname, args, func="cosine", path_to_plots=folder_path):
		assert all(list(map(lambda x : x >= -1 and x <= 1, scores)))
		assert dataname == "epilepsy"
		assert np.shape(args["X"])[1] == 10 
		from constants import dr_folder
		file_="rewards_cosine_10drugs_18samples"
		self.scores = list(map(float, scores))
		self.names =  list(map(str, args["names"]))
		assert len(self.names) == 10
		self.arm_ids, self.ids = get_drug_id(self.names, dr_folder+file_+".txt")
		assert not any([str(s) == "None" for s in self.arm_ids])
		self.rewards = np.loadtxt(dr_folder+file_+".csv")
		assert all([key in list(args.keys()) for key in ["X", "grn_name"]])
		## drug signatures
		# non binarized (for baseline)
		self.X = args["X"]
		## quantile normalized phenotype of patients
		patients = pd.read_csv(dr_folder+"GSE77578_patients.csv")
		patients.index = patients[patients.columns[0]]
		patients = patients.drop(columns=[patients.columns[0]])
		patients = patients.loc[~patients.index.duplicated()]
		self.patients = patients
		self.args = {"X": self.X.values}
		self.type = "gaussian"
		assert is_of_type_LIST(self.scores, "float")
		assert is_of_type_OPTION(self.args, "dict")
		assert "X" in list(self.args.keys())
		N, K = np.shape(self.args["X"])
		self.scores_filepath = path_to_plots+"scores_"+self.type+"_"+dataname+"_K="+str(K)+"_N="+str(N)+".csv"
		self.X_filepath = path_to_plots+"features_"+self.type+"_"+dataname+"_K="+str(K)+"_N="+str(N)+".csv"
		self.oracle = self.scores
		with open(self.scores_filepath, "w") as writeFile:
			writer = csv.writer(writeFile)
			writer.writerows([[s] for s in self.oracle])
		np.savetxt(self.X_filepath, args["X"])

	def get_reward(self, arm, quiet=True, sample_id=None):
		if (not is_of_type(sample_id, "int")):
			from random import sample
			sample_id = sample(range(np.shape(self.patients.values)[1]), 1)[0]
		score = float(self.rewards[sample_id, self.ids[arm]])
		if (not quiet):
			print(("Arm: ", (arm, self.arm_ids[arm], self.names[arm]), "predicted, true scores = ", (score, self.scores[arm])))
		return score

####################################################
## Factory                                        ##
####################################################

#' @param problem Python character string
#' @param scores Python float list
#' @param dataname Python character string
#' @param args Python dictionary
#' @return problem custom GenericProblem class object: generator of arm rewards
def problem_factory(problem, scores, dataname, args, path_to_plots):
	'''Factory for problems: returns a GenericProblem instance initialized on @scores values and additional arguments stored in @args'''
	assert is_of_type_LIST(scores, "float")
	assert is_of_type(args, "dict")
	assert is_of_type(problem, "str")
	di = {
		"bernouilli": (lambda _ : BernouilliProblem),
		"gaussian": (lambda _ : GaussianProblem),  
		"poisson": (lambda _ : PoissonProblem), 
		"exponential": (lambda _ : ExponentialProblem), 
		"epilepsy": (lambda _ : DRProblem), 
		"epilepsySubset": (lambda _ : DRProblemSubset), 
	}
	if (not (problem in list(di.keys()))):
		print("\""+problem+"\" not in "+str(list(di.keys())))
		raise ValueError
	return di[problem](0)(scores, dataname, args, path_to_plots=path_to_plots)
