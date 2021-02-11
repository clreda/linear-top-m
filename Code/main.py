# coding: utf-8

## Command
# default arguments in args.json
# CLASSIC: ```python main.py --small_K <number of arms> --beta <beta> --data classic --omega <angle> --bandit <bandit> --m <m> --is_greedy <1|0 ...```
# LINEAR: ```python main.py --small_K <number of arms> --small_N <number of dimensions> --beta <beta> --data linear --bandit <bandit> --vr <vr> --m <m> --is_greedy <1|0> ...```
# LOGISTIC: ```python main.py --small_K <number of arms> --small_N <number of dimensions> --beta <beta> --data logistic --bandit <bandit> --vr <vr> --m <m> --is_greedy <1|0> ...```
# DR: ```python main.py --small_K <number of arms> --beta <beta> --data epilepsy --bandit <bandit> --m <m> --is_greedy <1|0> ...```

from time import time
import numpy as np
import argparse
import random
import subprocess as sb
import csv
import yaml
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

import bandits
import utils
import betas
import problems
import data
from data import data_list

from constants import target_folder, folder_path

# For replication
random.seed(123456)

parser = argparse.ArgumentParser(description='Linear Top-m identification Bandit Algorithms & Drug Repurposing')
parser.add_argument('--data', type=str, 
                    help='DATASETS:\n- "epilepsy" for drug repurposing instance;\n- "linear" for normalized randomly sampled feature matrix and linear coefficient for scores;\n- "logistic" for normalized randomly sampled feature matrix and logistic coefficient for scores;\n- "classic" for KD bandit instance where m arms are orthogonal and another one is close to one of the first two but suboptimal.')
parser.add_argument('--bandit', type=str,
                    help='Name of the bandit to be tested: defined in "bandits.py".')
parser.add_argument('--json_file', type=str, default="args.json",
                    help='Path to JSON file with default value of arguments.')
parser.add_argument('--problem', type=str,
                    help='Name of one of the problem types defined in "problems.py".')
parser.add_argument('--beta', type=str,
                    help='Name of one of the exploration rate types defined in "betas.py".')
parser.add_argument('--m', type=int,
                    help='Number of best arms that should be returned in the related (epsilon, delta)-EXPLORE-m problem.')
parser.add_argument('--delta', type=float, 
                    help='Value of delta parameter in the related (epsilon, delta)-EXPLORE-m problem.')
parser.add_argument('--epsilon', type=float, 
                    help='Value of epsilon parameter in the related (epsilon, delta)-EXPLORE-m problem.')
parser.add_argument('--sigma', type=float,
                    help='Value of sigma parameter (~ noise in rewards).')
parser.add_argument('--n_simu', type=int,
                    help='Number of iterations of the algorithm.')
parser.add_argument('--mode', type=str, choices=['finetuning', 'recommendation', 'small_test', 'clear', 'generate_latex'],
                    help='Test to be performed:\n- "recommendation" which will give recommendations;\n- "finetuning" where the value of parameter which name is given with the "parameter" argument, start value "start", end value "end", step "step", is modified and the difference in empirical sample complexity and error frequency is recorded, and values are returned sorted by their performance;\n- "clear" to remove *.pyc files;\n- "generate_latex" to generate corresponding LateX source file.')
parser.add_argument('--verbose', type=int, choices=range(2), default=0,
                    help='Verbose during learning rounds.')
parser.add_argument('--plot', type=int, choices=range(2), default=1,
                    help='Plots figure of sample complexity, performance, distance to "true" theta and bandit instance.')
parser.add_argument('--plot_rounds', type=int, choices=range(3), default=0, 
                    help='Plots rounds "0": don\'t plot "1": plot first round "2": plot all rounds.')
parser.add_argument('--plot_step', type=int, 
                    help='[plot_rounds>0] Plot rounds every @plot_step rounds.')
################# FINETUNING ###############################################
parser.add_argument('--parameter', type=str, default="alpha",
                    help='[mode="finetuning"] Parameter to finetune.')
parser.add_argument('--start', type=float, 
                    help='[mode="finetuning"] Starting value for finetuning.')
parser.add_argument('--end', type=float, 
                    help='[mode="finetuning"] Ending value for finetuning.')
parser.add_argument('--step', type=float, 
                    help='[mode="finetuning"] Step value for finetuning.')
################# DATA       ###############################################
parser.add_argument('--small_K', type=int, 
                    help='[mode="small_test" | data in "data.py"] Value of K (number of arms).')
parser.add_argument('--small_N', type=int, 
                    help='[data in "data.py"] Value of N (number of dimensions).')
parser.add_argument('--omega', type=str,
                    help='[data="classic"] If data="classic": proximity of third suboptimal arm to the optimal arm (e.g. "1pi/3", "2pi/3" and as a general rule "[num]pi/[denom]").')
parser.add_argument('--vr', type=float, default=3.,
                    help='[data="linear"] If data="linear": value of standard deviation of Gaussian distribution from which features are sampled.')
parser.add_argument('--grn_name', type=str, 
                    help='Filename for GRN model file.')
parser.add_argument('--path_to_grn', type=str, 
                    help='Relative/absolute path to the GRN solver.')
################# BANDIT     ###############################################
parser.add_argument('--T_init', type=int,
                    help='[bandit=TrueUniform] Number of uniform arm samplings.')
parser.add_argument('--use_chernoff', type=str, choices=["none", "gaussian", "bernouilli"], default="None",
                    help='[bandit=GIFA] Using Chernoff stopping rule (/!\ theoretically valid only for m=1, epsilon=0).')
parser.add_argument('--eta', type=float, 
                    help='Custom eta value (~ variance of arm features = lambda/sigma in the paper).')
parser.add_argument('--k1_diff', type=float,
                    help='[bandit=KL-LUCB] Custom finetuning of alpha parameter (~ threshold function).')
parser.add_argument('--use_tracking', type=str, choices=["C", "D", "ForcedExplorationD", "ForcedExplorationC"], default="D",
                    help='[bandit=LinGame] Type of tracking rule ~ sampling.')
parser.add_argument('--alpha', type=float,
                    help='[bandit=LinLUCB|LUCB|KL-LUCB] Value of alpha parameter (~ threshold function).')
parser.add_argument('--is_greedy', type=int,
                    help='[bandit=LinLUCB|LinGapE|LinUGapE|LinIAA|LinGIFA] If one should sample a third "most informative arm" (like in LinGapE) rather than one of the challenger or best guess. [bandit=LUCB] If one should pull the least sampled arm between the best arm and challenger, instead of the original rule which samples both.')
args = parser.parse_args()

## Retrieve default values from JSON file: values provided via the command line have a higher priority than the ones in the JSON file
if (os.path.exists(args.json_file)):
	class Bunch(object):
		def __init__(self, adict):
			self.__dict__.update(adict)
	args_ = yaml.safe_load(open(args.json_file, "r").read())
	for a in list(vars(args).keys()):
		if (str(vars(args)[a]) != "None"):
			v = vars(args)[a]
			if (a in list(args_.keys())):
				args_[a] = v
			else:
				args_.setdefault(a, v)
	args = Bunch(args_)

args.verbose = bool(args.verbose)
args.plot = bool(args.plot)
args.plot_rounds = int(args.plot_rounds)
args.is_greedy = bool(args.is_greedy)
if (args.data not in ["linear", "classic", "logistic"]):
	args.problem = args.data

if (not os.path.isdir(folder_path)):
	sb.call("mkdir "+folder_path, shell=True)

## Parsing the value of omega
omega_str = ""
if (args.data == "classic"):
	args.omega, omega_str = utils.parse_omega_str(args.omega)

## Define folder and file names associated with each experiment

general_args = ["data", "bandit", "problem", "m", "n_simu", "verbose", "plot"]
if (args.mode in ["small_test", "recommendation", "generate_latex"]):
	path_to_plots = target_folder+args.data+("_"+omega_str if (args.data == "classic") else "")
	path_to_plots += ("_K"+str(args.small_K) if (args.data != "epilepsy" and ((args.mode == "small_test" or not utils.is_of_type(args.small_K, "NoneType")) and args.data not in ["classic", "epilepsy"]) or args.data in data_list) else "")
	path_to_plots += ("_N"+str(args.small_N) if (args.data in data_list and not args.data == "classic") else "")
	path_to_plots += "_"+args.problem+("_m="+str(args.m) if (args.m) else "")
	path_to_plots += "_delta="+str(args.delta)
	path_to_plots += "_epsilon="+str(args.epsilon)
	path_to_plots += "/"
	if (not os.path.isdir(path_to_plots)):
		sb.call("mkdir "+path_to_plots, shell=True)
	## Save parameters
	with open(path_to_plots+"parameters.json", "w") as f:
		f.write(str(vars(args)))
	import sys
	## Save command-line call
	with open(path_to_plots+"last_command.txt", "w") as f:
		cmd = "python "+reduce(lambda x,y : x+" "+y, sys.argv)
		f.write(cmd)

####################################################
##  INITIALIZE BANDITS & ARMS                     ##
####################################################

#' @params X NumPy matrix of size N x K
#' @param data Python character string
#' @param T_init Python integer 0 < T_init
#' @param m Python integer 0 < m < K
#' @param sigma Python float 0 < sigma
#' @param alpha Python float 1 < alpha
#' @param eta Python float 0 < eta
#' @param k1_diff Python float 0 < k1_diff
#' @param epsilon Python float 0 <= epsilon
#' @param delta Python float 0 < delta
#' @param verbose Python bool
#' @return bandit custom ExploreMBandit instance for the currently selected parameter values
def select_bandit(X, df_di, oracle, data=args.data, theta=None, T_init=args.T_init, m=args.m, sigma=args.sigma, alpha=args.alpha, eta=args.eta, k1_diff=args.k1_diff, is_greedy=args.is_greedy, plot_step=args.plot_step, epsilon=args.epsilon, delta=args.delta, verbose=args.verbose, use_tracking=args.use_tracking, print_test=True):
	'''Creates the bandit instance associated with feature matrix @X, oracle scores @oracle, wrt. data @data, with bandit-specific parameters, for the (@epsilon, @delta)-PAC EXPLORE-m problem'''
	assert args.bandit
	assert utils.is_of_type(X, "numpy.matrix")
	N, K = np.shape(X)
	assert utils.is_of_type_LIST(oracle, "float")
	assert len(oracle) == K
	assert utils.is_of_type(data, "str")
	assert m > 0 and m < K
	assert utils.is_of_type(sigma, "float")
	assert utils.is_of_type(alpha, "float")
	assert alpha > 1
	assert utils.is_of_type(eta, "float")
	assert eta > 0
	assert utils.is_of_type(k1_diff, "float")
	assert k1_diff > 0
	assert utils.is_of_type(is_greedy, "bool")
	assert utils.is_of_type(epsilon, "float")
	assert epsilon >= 0
	assert utils.is_of_type(delta, "float")
	assert delta > 0
	assert utils.is_of_type(verbose, "bool")
	assert not utils.is_of_type(theta, "NoneType")
	S = float(np.linalg.norm(theta, 2))
	assert utils.is_of_type(S, "float")
	assert args.beta
	assert S > 0
	args_problem = {"sigma": sigma, "grn_name": args.grn_name, "path_to_grn": args.path_to_grn}
	## DR instances use data frames in order to match genes properly
	if (args.data in data_list):
		args_problem.update({"X": X, "S": None})
	else:
		args_problem.update(df_di)
	aproblem = args.problem
	if (args.problem == "epilepsy" and args.small_K == 10):
		aproblem += "Subset"
	## Select problem object
	problem = problems.problem_factory(aproblem, oracle, data, args_problem, path_to_plots)
	## Select threshold function
	beta = betas.beta_factory(args.beta, {"delta": delta, "alpha": alpha, "X": X, "sigma" : sigma, "k1_diff": k1_diff, "eta": eta, "S": S})
	## Annotate the experiment
	params="_m="+str(m)+"_delta="+str(delta)+"_epsilon="+str(epsilon)+"_problem="+aproblem+"_sigma="+str(sigma)
	if (args.data != "epilepsy"):
		params+="_K="+str(args.small_K)
	params+="_alpha="+str(alpha)+"_eta="+str(eta)
	params+="_k1_diff="+str(k1_diff)
	params+="_data="+data+"_beta="+args.beta
	## Automatically perform finetuning on the T_unit parameter when the chosen bandit algorithm is TrueUniform without a provided value of T_unit
	if (utils.is_of_type(T_init, "NoneType") and args.bandit == "TrueUniform"):
		start, step = 10 if (not args.start) else int(args.start), 10 if (not args.step) else int(args.step)
		end, n_simu = 100 if (not args.end) else int(args.end), args.n_simu
		print("Finetuning T_init in seq("+str(start)+", "+str(end)+", "+str(step)+") across "+str(n_simu)+" simulations "),
		trueunif = bandits.bandit_factory("TrueUniform", {"T_init": 1, "plot_name": args.bandit}, X, m, problem, theta, "feature", delta, epsilon, verbose, False, params, path_to_plots=path_to_plots, plot_step=plot_step)
		mat = trueunif.grid_search("T_init", start, step, end, data_name=data, n_simu=n_simu, get_plot=False)
		pes = mat[:,2]
		## argmax = first (thus smallest) index such that the correctness frequency is maximal
		T_init = range(start, end+step+1, step)[np.argmax(pes.T)]
		print("T_init = "+str(T_init))
		assert utils.is_of_type(T_init, "int")
		assert T_init > 0
	params+="_T_init="+str(T_init)
	args_ = {"beta": beta, "plot_name": args.bandit+"_"+args.beta, "sigma": sigma, "alpha": alpha, "eta": eta, "k1_diff":k1_diff, "T_init": T_init, "is_greedy": is_greedy, "use_chernoff": args.use_chernoff, "use_tracking": use_tracking}
	bandit = bandits.bandit_factory(args.bandit, args_, X, m, problem, theta, "feature", delta, epsilon, verbose, False, params, path_to_plots, plot_step)
	return bandit

## INITIALIZE BANDIT
if (not (args.mode in ["clear"])):
	assert not (utils.is_of_type(args.data, 'NoneType'))
	X, oracle, theta, names, df_di = data.create_scores_features(args, folder_path)
	if (not (args.mode in ["small_test"])):
		assert args.m
		m = args.m

####################################################
##  BENCHMARK                                     ##
####################################################

## More readable summary of results for arms
def print_names(bandit, min_list=5, ndec=5):
	try:
		names = bandit.problem.names
	except:
		names = None
	if (not utils.is_of_type(bandit.empirical_recommendation, "NoneType") and not utils.is_of_type(names, "NoneType")):
		m = min(min_list, bandit.m)
		ids = utils.m_maximal(list(map(float,bandit.empirical_recommendation.tolist())), m)
		true_ids = utils.m_maximal(list(map(float,bandit.problem.oracle)), m)
		output = ""
		output += "-- Names of the first "+str(m)+" most recommended items across all "+str(args.n_simu)+" simulations:\n"
		output += str([bandit.problem.names[i] for i in ids])+"\n"
		output += "-- Associated (empirical) scores:\n"
		output += str(list(map(lambda x : round(x, ndec), [bandit.empirical_means[i] for i in ids])))+"\n"
		output += "-- Names of the first "+str(bandit.m)+" 'truly good' items:\n"
		output += str([bandit.problem.names[i] for i in true_ids])+"\n"
		output += "-- Associated (true) scores:\n"
		output += str([round(bandit.problem.oracle[i], ndec) for i in true_ids])+"\n"
		print(output)
		with open(bandit.path_to_plots+bandit.name+"_recommendation_eps="+str(args.epsilon)+"_beta="+args.beta+".txt", "w") as f:
			f.write(output)
		return output
	else:
		if (not utils.is_of_type(names, "NoneType")):
			assert utils.is_of_type(bandit.m, "int")
			assert utils.is_of_type_LIST(bandit.problem.oracle, "float")
			true_ids = utils.m_maximal(list(map(float,bandit.problem.oracle)), bandit.m)
			output = "-- Names of the first "+str(bandit.m)+" 'truly good' items:\n"
			output += str([bandit.problem.names[i] for i in true_ids])+"\n"
			output += "-- Associated (true) scores:\n"
			output += str([round(bandit.problem.oracle[i], ndec) for i in true_ids])+"\n"
			return output

################# FINETUNING ###############################################
if (args.mode == "finetuning"):
	bandit = select_bandit(X, df_di, oracle, args.data, m=m, theta=theta)
	print("Finetune parameter '"+args.parameter+"' seq("+str(args.start)+", "+str(args.step)+", "+str(args.end)+")")
	assert args.bandit
	assert args.n_simu
	assert args.data
	assert args.parameter
	assert args.start
	assert args.end
	assert args.step
	plot_mat = bandit.grid_search(args.parameter, args.start, args.step, args.end, n_simu=args.n_simu, get_plot=args.plot, data_name=args.data)
	print("Best values of "+args.parameter+" for improved sample complexity:")
	get_ids = lambda X : np.argsort(X.flatten().tolist()).tolist()[0]
	ids = get_ids(plot_mat[:, 1])
	print([plot_mat[i, 0] for i in ids])
	print("Ordered values of "+args.parameter+" for improved error frequency:")
	ids = list(reversed(get_ids(plot_mat[:, 2])))
	print([plot_mat[i, 0] for i in ids])
	print("Value(s) of "+args.parameter+" for minimum error frequency:")
	ids = np.argwhere(np.array(plot_mat[:, 2].flatten().tolist()[0]) == np.max(plot_mat[:, 2])).flatten().tolist()
	print([plot_mat[i, 0] for i in ids])
	if (not utils.is_of_type(bandit.theta, "NoneType")):
		print("-- Distance to true theta")
		print(np.round(np.linalg.norm(bandit.true_theta-bandit.theta, 2), 3))
################# TESTING (REC) ############################################
## Meaning we run the bandit on the full dataset (for real-life data)
elif (args.mode == "recommendation"):
	assert args.bandit
	assert args.n_simu
	bandit = select_bandit(X, df_di, oracle, args.data, m=m, theta=theta)
	bandit.plot_results(n_simu = args.n_simu, show_plot=args.plot, plot_rounds=args.plot_rounds)
	if (args.plot):
		plt.figure(figsize=(20, 4.19))
		utils.plot_instance(bandit)
	print("* Recommendation")
	print(sorted(bandit.recommend()[0]))
	print("* Oracle")
	print(utils.m_maximal(bandit.problem.oracle, bandit.m))
	print("* Means")
	print("-- Estimated means of recommended arms")
	print(np.round(bandit.recommend()[1], 2))
	print("-- Estimated means of oracle arms")
	print(np.round([bandit.means[i] for i in utils.m_maximal(bandit.problem.oracle, bandit.m)], 2))
	if (not utils.is_of_type(names, "NoneType")):
		print_names(bandit)
################# TESTING    ###############################################
## Meaning we select/generate a smaller dataset
elif (args.mode in ["small_test"]):
	assert args.small_K
	assert args.n_simu
	N, K = np.shape(X)
	k = min(args.small_K,K)
	Xprime_filepath = path_to_plots+"features_"+args.problem+"_"+args.data+"_K="+str(k)+"_N="+str(N)+".csv"
	scores_filepath = path_to_plots+"scores_"+args.problem+"_"+args.data+"_K="+str(k)+"_N="+str(N)+".csv"
	if (not os.path.exists(Xprime_filepath) or not os.path.exists(scores_filepath)):
		print("Building new subproblem K="+str(k)+" and N <= "+str(N)+"...")
		from random import sample
		# Choose a random subset of size k in [K]
		test_idx = sample(range(K), k)
		if (not args.data in data_list):
			with open(path_to_plots+"selected_item_ids.txt", "w") as f:
				f.write("Selected ids (size="+str(k)+") among "+str(K)+"elements in data \""+args.data+"\":\n"+str(test_idx))
		Xprime = X[:, test_idx]
		oracle = [oracle[i] for i in test_idx]
		if (len(df_di) > 0):
			X, S = [df_di[col][df_di[col].columns[test_idx]] for col in ["X", "S"]]
			df_diprime = {"X": X, "S": S, "names": [df_di["names"][i] for i in test_idx]}
		else:
			df_diprime = {}
	else:
		print("Loading existing subproblem K="+str(k)+" and N="+str(N)+"...")
		Xprime = np.matrix(np.loadtxt(Xprime_filepath))
		oracle = np.loadtxt(scores_filepath).tolist()
		assert utils.is_of_type_LIST(oracle, "float")
		assert utils.is_of_type(Xprime, "numpy.matrix")
		if (len(df_di) > 0):
			with open(path_to_plots+"selected_item_ids.txt", "r") as f:
				lines = f.read().split("\n")
				test_idx = list(map(int,lines[1][1:-1].split(",")))
			X, S = [df_di[col][df_di[col].columns[test_idx]] for col in ["X", "S"]]
			df_diprime = {"X": X, "S": S, "names": [df_di["names"][i] for i in test_idx]}
		else:
			df_diprime = {}
	assert args.m
	m = args.m
	bandit = select_bandit(Xprime, df_diprime, oracle, args.data, m=m, theta=theta, print_test=args.verbose)
	if (bandit.compute_hardness()[1] > 1e10):
		raise ValueError("Instance is too hard")
	if (args.plot):
		plt.figure(figsize=(20, 4.19))
		utils.plot_instance(bandit)
	bandit.plot_results(n_simu=args.n_simu, show_plot=args.plot, plot_rounds=args.plot_rounds)
	plot_mat = np.zeros((2, bandit.t), dtype=np.int)
	plot_mat[0,:] = [i for i in bandit.rewards]
	plot_mat[1,:] = bandit.pulled_arms
	## Print which arms have been sampled for short runs
	if (np.shape(plot_mat)[1] < 20):
		print("\n---------\n")
		for i in range(np.shape(plot_mat)[1]):
			print("playing arm "+str(plot_mat[1, i])+" yielded reward "+str(plot_mat[0, i]))
	if (not utils.is_of_type(bandit, "NoneType")):
		print_names(bandit)
################# CLEAN       ###############################################
## Clear all files (and all experiments!)
elif (args.mode == "clear"):
	sb.call("rm -f *.csv", shell=True)
	sb.call("rm -rf "+target_folder+" && mkdir "+target_folder, shell=True)
################# LATEX       ###############################################
## Generate automatically a PDF file with the sample complexity, error frequency and recommendation results
## along with a boxplot which compares the sample complexity across algorithms for the specified experiment
elif (args.mode == "generate_latex"):
	N, K = np.shape(X)
	if (args.small_K):
		K = min(args.small_K,K)
	utils.generate_latex_file(path_to_plots, args.data, K, N, m, args.problem, omega_str)
	utils.build_boxplot_per_experiment(path_to_plots, args.data, K, N, m, args.problem, omega_str)
else:
	print("You need to choose a mode: choices=['finetuning', 'recommendation', 'small_test', 'clear', 'generate_latex'].")

## Cleaning up Python import files
sb.call("rm -f *.pyc", shell=True)
