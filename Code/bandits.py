# coding: utf-8

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from random import sample
import subprocess as sb
import os
import gc

import utils
from indices import PairedContextualIndex, DisjointContextualIndex, DisjointNonContextualIndex, Index
from learners import AdaHedge, FixedShare, FTL

####################################################
## General BANDIT class for the EXPLORE-m problem ##
####################################################

#' @param args Python dictionary
#' @param X NumPy matrix
#' @param m Python integer (0 < m < np.shape(X)[1])
#' @param problem custom GenericProblem instance (defined in "utils.py")
#' @param delta Python float (0 < delta < 1)
#' @param epsilon Python float (0 <= epsilon)
#' @param verbose Python bool
#' @param warning Python bool
#' @param params Python character string
#' @return bandit custom ExploreMBandit instance
class ExploreMBandit(object):
	'''[Not implemented] Returns a ExploreMBandit bandit instance that solves the associated (@epsilon, @delta)-EXPLORE-@m problem with feature matrix @X, with arms defined in @problem, and bandit-specific arguments in @args. @X_type is the type of X ("feature": feature matrix, "correlation": correlation matrix)'''
	def __init__(self, args, X, m, problem, theta=None, X_type="feature", delta=0.05, epsilon=0., verbose=False, warning=False, params="", path_to_plots="../Results/", plot_step=100):
		assert utils.is_of_type(delta, "float") and delta > 0 and delta < 1
		assert utils.is_of_type(epsilon, "float") and epsilon >= 0
		assert utils.is_of_type_LIST([verbose, warning], "bool")
		assert utils.is_of_type(args, "dict")
		assert utils.is_of_type(X, "numpy.matrix")
		assert np.shape(X)[0] > 0 and np.shape(X)[1] > 0
		assert utils.is_of_type(m, "int") and m > 0 and m < np.shape(X)[1]
		assert utils.is_of_type(params, 'str')
		assert utils.is_of_type(X_type, "str") and X_type in ["feature", "correlation"]
		assert utils.is_of_type(plot_step, "int") and plot_step > 0
		self.name = "ExploreMBandit"
		self.not_terminated = 0
		self.current_n_simu = 0
		self.verbose = verbose
		self.warning = warning
		self.delta = delta
		self.epsilon = epsilon
		self.plot_step = plot_step
		self.m = m
		self.X = X
		self.N, self.K = np.shape(X)
		self.arms = range(self.K)
		self.problem = problem
		self.params = params
		self.true_theta = theta
		self.theta = None
		self.X_type = X_type
		self.path_to_plots = path_to_plots
		self.T_init = 0
		## Bound on l2-norm of theta
		if (not utils.is_of_type(theta, "NoneType")):
			self.S = float(np.linalg.norm(theta, 2))
		else:
			self.S = 1 #assuming theta ("contributions of dimension to arm reward") is normalized
		## Bound on l2-norm of feature vectors
		self.L = float(np.max(np.sum(np.abs(np.multiply(self.X, self.X)), axis=0))**(1./2))
		for attr in list(args.keys()):
			setattr(self, attr, args[attr])
		assert utils.is_of_type(self.T_init, 'int')
		assert utils.is_of_type_OPTION(self.problem.oracle, 'list')
		if (utils.is_of_type(self.problem.oracle, "list")):
			assert utils.is_of_type_LIST(self.problem.oracle, "float")
			assert len(self.problem.oracle) == self.K
		self.clear()
		try:
			self.plot_name
		except:
			self.plot_name = self.name
		self.problem_info = self.path_to_plots+"problem.txt"
		from glob import glob
		if (len(glob(self.path_to_plots+"problem_*.txt")) == 0):
			arms = utils.m_maximal(self.problem.oracle, self.K)
			means = np.sort([round(x, 3) for x in self.problem.oracle]).tolist()
			means.reverse()
			csts = self.compute_hardness()
			txt = "gap,hardness\n"
			txt += str(round(csts[2], 3))+","+str(csts[0])+"\n"
			with open(self.problem_info, "w") as f:
				f.write(txt)
			plt.figure(figsize=(20, 4.19))
			plt.hist(self.problem.oracle, bins=100, density=False, label="score")
			plt.savefig(self.problem_info[:-4]+'_scores.png', bbox_inches='tight')
		H = self.compute_hardness()[1]
		max_iter_pow = len(str(H))
		max_iter = int(0.1**(max_iter_pow-1)*H)+1
		self.max_it = max_iter*10**(max_iter_pow-1)
		self.max_it = int(max(min(1e7, self.max_it), 1e5))
		print("/!\ Non converging run bound = "+str(self.max_it)+" iterations")
		## "break ties" functions
		self.randf = utils.randf

	def clear(self):
		## Collected rewards
		self.rewards = []
		## Pulled arms
		self.pulled_arms = []
		## Should the learning step stop?
		self.done = False
		self.t = 0
		## Estimated empirical average reward for each arm
		self.means = np.zeros(self.K)
		self.cum_sum = [0]*self.K
		## Contextual algorithms
		self.B_inv = None
		self.b = None
		self.theta = None
		## Number of times each arm has been sampled so far
		self.na = np.zeros(self.K)
		## Plot arguments
		## Confidence intervals (if needed)
		self.confidence_intervals = None
		self.best_arm, self.challenger = [None]*2
		## Stopping quantity
		self.B = float("inf")
		self.indices = None
		## "optimal" allocation
		self.ratio = {}
		self.reset_bandit_specific_parameters()

	#' @param arm Python integer (arm id) in 0...K-1
	def play(self, arm):
		assert utils.is_of_type(arm, 'int') and arm in self.arms
		reward = self.problem.get_reward(arm)
		if (self.verbose):
			print("Playing arm " + str(arm) + " yields reward " + str(reward))
		assert (utils.is_of_type(reward, 'int') or utils.is_of_type(reward, 'float'))
		self.rewards.append(reward)
		self.pulled_arms.append(arm)

	#' @param candidates Python integer list (arm id) in 0...K-1
	def plot_current_round(self, candidates):
		step = self.plot_step
		assert utils.is_of_type(step, "int") and step > 0
		round_path=self.path_to_plots+"Rounds_"+self.plot_name+"_"+self.params+"_nsimulation="+str(self.current_n_simu)+"/"
		if (not os.path.isdir(round_path)):
			sb.call("mkdir "+round_path, shell=True)
		if (self.t % step in range(2) or self.stopping_rule()):
			assert utils.is_of_type(self.B, "float")
			utils.plot_confidence_intervals(self.m, self.epsilon, candidates, self.confidence_intervals, self.means, self.na, self.plot_name, self.B, oracle=self.problem.oracle, best_arm=self.best_arm, challenger=self.challenger, indices=self.indices, mean_bound=self.L*self.S)
			plt.savefig(round_path+'t='+str(self.t)+'.png', bbox_inches='tight')
			plt.close()

	#' @param candidates Python integer list (arm id) in 0...K-1
	#' @param plot_rounds Python boolean
	def learn_from_arms(self, candidates, plot_rounds):
		if (len(candidates) > 0):
			assert utils.is_of_type_LIST(candidates, "int")
			if (len(candidates) > 0):
				if (plot_rounds):
					self.plot_current_round(candidates)
				for arm in candidates:
					self.play(arm)
					self.t += 1
					self.na[arm] += 1
				self.update(candidates)

	## Uniformly explore each arm in arms T_init times before the learning step
	#' @param plot_rounds Python boolean
	def initialization(self, plot_rounds, T_init=None):
		if (str(T_init) == None):
			T_init = self.T_init
		candidates = [[arm]*T_init for arm in self.arms]
		candidates = [y for x in candidates for y in x]
		assert len(candidates) == T_init*self.K
		self.learn_from_arms(candidates, plot_rounds)

	#' @returns idx, means the arm ids of the recommended set, associated means (in the order of arm ids)
	def recommend(self):
		if (self.verbose):
			print("--- End of learning step")
		idx = self.recommend_rule()
		idx = list(map(int, idx))
		means = self.means[idx].tolist()
		assert utils.is_of_type_LIST(idx, "int") and all([i in self.arms for i in idx]) and len(idx) == self.m
		assert utils.is_of_type_LIST(means, "float") and len(means) == self.m
		return idx, means

	## Run the entire algorithm
	#' @param plot_rounds Python boolean
	#' @param verbatim Python boolean
	#' @returns result, nb result contains recommended set (ids+means), nb is the number of sampled arms
	def run(self, plot_rounds=True, verbatim=False):
		self.clear()
		self.initialization(plot_rounds, T_init=self.T_init)
		i = 0
		while not self.done:
			if (verbatim):
				print("** Round #"+str(i+1)),
			i += 1
			candidates = self.sample()
			self.learn_from_arms(candidates, plot_rounds)
			self.done = self.stopping_rule()
			if (i > self.max_it):
				print("\n\nRun #"+str(self.current_n_simu)+" has not terminated (nrounds > "+str(i)+")\n\n")
				self.not_terminated += 1
				break
		if (plot_rounds):
			self.plot_current_round(self.sample())
		## FIRST element of result is always the recommended arm indices
		## other elements are lists with information on recommended arms
		result = self.recommend()
		return result, len(self.pulled_arms)

	## UGapE complexity constant
	#' @returns hardness, H, gap: sample complexity upper bound from UGapE, H associated complexity constant, gap mu[m]-mu[m+1] means of extreme arms
	def compute_hardness(self):
		if (not utils.is_of_type(self.problem.oracle, 'NoneType')):
			oracle_ = np.sort(self.problem.oracle).tolist()
			oracle_.reverse()
			gaps = [oracle_[i]-oracle_[self.m-1] for i in range(self.m-1)]
			gaps += [oracle_[self.m-1]-oracle_[i] for i in range(self.m,self.K)]
			gaps = np.array(gaps)
			gaps = gaps[gaps > 0]
			gaps = gaps.tolist()
			gap = oracle_[self.m-1]-oracle_[self.m]
			H = sum(list(map(lambda x : 1./float(max(self.epsilon, (x+self.epsilon)*0.5)**2), gaps)))
			hardness = H*np.log(H/self.delta)
			return int(hardness), H, gap
		return None

	## Empirical sample complexity & Error frequency
	## if oracle exists, whether the algorithm returns a good answer
	## /!\ executes n_simu times the algorithm!
	#' @param n_simu number of simulations of the algorithm
	#' @param show_plot show the plot
	#' @returns sc, pe sc: average number of samplings across n_simu iterations, pe: average correctness across n_simu iterations
	def plot_results(self, n_simu=10, show_plot=True, plot_rounds=1, verbatim=False, verbatim2=True, ndec=5):
		import time
		iterations = range(n_simu)
		sample_complexity, is_good_answer, total_time = [], [], []
		prop_drawing_arms = np.zeros(self.K)
		self.empirical_means = np.zeros(self.K)
		self.empirical_recommendation = np.zeros(self.K)
		if (verbatim2):
			fname2 = self.path_to_plots+self.plot_name+'.txt'
			with open(fname2, "w+") as f:
				f.write("simulation,sample_comp_mean,sample_comp_std,correctness"+(",theta_dist" if (not utils.is_of_type(self.theta, "NoneType")) else "")+"\n")
			## Human readable version
			fname = self.path_to_plots+self.plot_name+'_hr.txt'
			if (os.path.exists(fname)):
				sb.call("rm -f "+fname, shell=True)
		for n in iterations:
			if (verbatim2):
				print("Simulation #" + str(n+1)+" ")
			self.current_n_simu = n+1
			## One run
			pltr=(n == 0) if (plot_rounds == 1) else (plot_rounds > 0)
			t = time.time()
			result, n_samples = self.run(plot_rounds=pltr, verbatim=verbatim)
			total_time.append(time.time()-t)
			gc.collect()
			result, means = result
			if (verbatim):
				print(":=> " + str(sorted(result)))
			sample_complexity.append(n_samples)
			self.empirical_recommendation[result] += 1
			for i in self.arms:
				prop_drawing_arms[i] += sum([i == x for x in self.pulled_arms])
			if (not utils.is_of_type(self.problem.oracle, 'NoneType')):
				is_good_test = [self.problem.oracle[i] >= utils.m_max(self.problem.oracle, self.m)-self.epsilon for i in result]
				is_good_answer.append(int(all(is_good_test)))
			self.empirical_means += self.means
			if (verbatim2):
				N = float(len(sample_complexity))
				print("Number of samplings: "+str(int(sample_complexity[-1])))
				sc = sum(sample_complexity)/N
				pe = sum(is_good_answer)/N
				output1 = "Simulation n="+str(n+1)+" (up to n):\n* SC = "+str(round(sc,3))+"\n* CF = "+str(round(pe*100, 5))+"%\n"
				output = output1+"* Empirical arm sampling = "+str((np.round(prop_drawing_arms/float(sum(sample_complexity)), ndec)).tolist())+"\n"
				output += "* Empirical recommendation = "+str((np.round(self.empirical_recommendation/N, ndec)).tolist())+"\n"
				if (not utils.is_of_type(self.theta, "NoneType")):
					output += "* Theta distance = "+str(np.round(np.linalg.norm(self.true_theta-self.theta, 2), ndec))+"\n"
				with open(fname, "a+") as f:
					f.write(output)
				with open(fname2, "a+") as f:
					f.write(str(n+1)+","+str(sc)+","+str(round(np.std(sample_complexity), ndec))+","+str(round(pe*100, ndec))+(","+str(np.round(np.linalg.norm(self.true_theta-self.theta, 2), ndec)) if (not utils.is_of_type(self.theta, "NoneType")) else "")+"\n")
				print(reduce(lambda x,y : x+" "+y, output1.split("\n")[1:]))
		N = float(n_simu)
		self.empirical_means /= N
		self.empirical_recommendation /= N
		sc = float(np.mean(sample_complexity))
		output = self.plot_name
		add_output = lambda output, x : output+"\n"+str(x)
		if (verbatim2):
			print("")
			mt = np.mean(total_time)
			time_lst = [mt//3600, (mt%3600)//60, (mt%3600)%60]
			time_lst = [x if x >= 1 else 0 for x in time_lst]
			output = add_output(output, "* Not terminated runs:")
			output = add_output(output, int(self.not_terminated))
			output = add_output(output, "* Average runtime (in sec.):")
			output = add_output(output, str(time_lst[0])+"H "+str(time_lst[1])+" min "+str(time_lst[2])+" sec ("+str(round(mt, 3))+" sec.)")
			output = add_output(output, "* Empirical recommendation:")
			output = add_output(output, np.round(self.empirical_recommendation,ndec))
			output = add_output(output, "* Arm drawing frequency:")
			output = add_output(output, np.round(prop_drawing_arms/float(sum(sample_complexity)),ndec))
			if (not utils.is_of_type(self.theta, "NoneType")):
				output = add_output(output, "-- Distance to true theta")
				output = add_output(output, np.round(np.linalg.norm(self.true_theta-self.theta, 2), ndec))
		if (verbatim2):
			output = add_output(output, "Average runtime (sec.) = " + str(round(sum(total_time)/N, ndec)))
			output = add_output(output, "Average sample number = " + str(round(sc,3)))
			output = add_output(output, "Std. in sample number = " + str(round(np.std(sample_complexity), 3)))
		if (not utils.is_of_type(self.problem.oracle, 'NoneType')):
			pe = sum(is_good_answer)/N	
			hardness, H, gap = self.compute_hardness()
			if (hardness == 0):
				hardness = 1
			if (min(hardness,sc) > 0):
				ratio = max(hardness,sc)/float(min(hardness,sc))
			else:
				ratio = float("inf")
			subbest = min(10, max(self.K-1, 1))
			if (verbatim2):
				output = add_output(output, "Expected sample complexity (UGapE) = O(" + str(hardness) + ")")
				output = add_output(output, "Oracle = " + str(sorted(self.problem.oracle)))
				output = add_output(output, "Gap best arms/worst arms (mu_{m} - mu_{m+1}) = " + str(round(gap, ndec)))		
				output = add_output(output, "Success frequency = " + str(pe) + ": {"+str(1-pe)+" <= "+str(self.delta)+"} = "+str(1-pe <= self.delta))
		if (verbatim2):
			output = add_output(output, "#simulations = "+str(n_simu))
			print(output)
			with open(self.path_to_plots+"output_"+self.plot_name+"_"+self.params+".txt", "w") as f:
				f.write(output)
		if (show_plot):
			tb = hardness if (not utils.is_of_type(self.problem.oracle, 'NoneType')) else None
			plt.figure(figsize=(20, 4.19))
			if (not utils.is_of_type(self.problem.oracle, 'NoneType')):
				plt.subplot(121)
				utils.plot_sample_complexity(sample_complexity, theoretical_bound=tb, xtitle='in '+self.plot_name+('\n(non term. runs: '+str(self.not_terminated)+')' if (self.not_terminated > 0) else ''))
				plt.subplot(122)
				utils.plot_performance(is_good_answer, xtitle='in '+self.plot_name+' (delta='+str(self.delta)+')')
			else:
				utils.plot_sample_complexity(sample_complexity, theoretical_bound=tb, xtitle='Simulation # in '+self.plot_name)
			plt.savefig(self.path_to_plots+self.plot_name+'.png', bbox_inches='tight')
		return sc, pe

	## Automated finetuning
	def grid_search(self, arg_name, start, step, end, data_name=None, n_simu=100, get_plot=True, ndec=5):
		int_args = ["T_init"]
		try:
			getattr(self, arg_name)
		except:
			print("'"+arg_name+"' not in arguments")
			raise ValueError
	 	print("\n-------- Testing values of '"+arg_name+"'")
		values = [(int if arg_name in int_args else float)(x) for x in range(start, end+step+1, step)]
		scs, pes, dtt = [], [], []
		params=values
		i = 0
		## Human readable
		fname = self.path_to_plots+self.plot_name+"_finetuning_"+arg_name+"_hr.txt"
		if (os.path.exists(fname)):
			sb.call("rm -f "+fname, shell=True)
		fname2 = self.path_to_plots+self.plot_name+"_finetuning_"+arg_name+".txt"
		with open(fname2, "w+") as f:
			f.write("test,value,sample_comp_mean,correctness\n")
		for value in values:
			print("\nTest #" + str(i+1) + "/"+str(len(values))+": "+arg_name+" = "+str(value))
			setattr(self, arg_name, value)
			assert (getattr(self, arg_name) == value)
			## Runs
			sc, pe = self.plot_results(n_simu = n_simu, show_plot=False, plot_rounds=0, verbatim2=False)
			scs.append(float(sc))
			pes.append(float(pe))
			if (not utils.is_of_type(self.theta, "NoneType")):
				dtt.append(float(np.linalg.norm(self.true_theta-self.theta, ndec)))
			if (True):
				output = "Test n="+str(i+1)+":\n ("+arg_name+" = "+str(value)+") * SC = "+str(round(sc, ndec))+"\n* CF = "+str(round(pe*100, ndec))+"%\n"
				with open(fname, "a+") as f:
					f.write(output)
				with open(fname2, "a+") as f:
					f.write(str(i+1)+","+str(value)+","+str(sc)+","+str(round(pe*100, ndec))+"\n")
				print(reduce(lambda x,y : x+" "+y, output.split("\n")[1:]))
			i += 1
		plt.figure(figsize=(20, 4.19))
		if (not utils.is_of_type(self.problem.oracle, 'NoneType')):
			if (not utils.is_of_type(self.theta, "NoneType")):
				plt.subplot(131)
				utils.plot_function(scs, arg_name, params, milestones=[self.compute_hardness()[0]], ms_dir="inf", xtitle='Sample complexity in function of '+arg_name+' in '+self.plot_name, col="b")
				plt.subplot(132)
				utils.plot_function(pes, arg_name, params, milestones=[1-self.delta, 1.], xtitle='Performance in function of '+arg_name+' in '+self.plot_name, col="r")
				plt.subplot(133)
				utils.plot_function(dtt, arg_name, params, milestones=[1e-3], ms_dir="inf", xtitle='Distance to true theta in function of '+arg_name+' in '+self.plot_name, col="g")
			else:
				plt.subplot(121)
				utils.plot_function(scs, arg_name, params, milestones=[self.compute_hardness()[0]], ms_dir="inf", xtitle='Sample complexity in function of '+arg_name+' in '+self.plot_name, col="b")
				plt.subplot(122)
				utils.plot_function(pes, arg_name, params, milestones=[1-self.delta, 1.], xtitle='Performance in function of '+arg_name+' in '+self.plot_name, col="r")
		else:
			utils.plot_function(scs, arg_name, params, milestones=[self.compute_hardness()[0]], ms_dir="inf", xtitle='Sample complexity in function of '+arg_name+' in '+self.plot_name, col="b")
		plt.savefig(self.path_to_plots+arg_name+'_finetuning.png', bbox_inches='tight')
		if (get_plot):
			plt.show()
		return np.matrix([params, scs, pes], dtype=float).transpose()

	#' @param C NumPy matrix of float of size K x K
	#' @return X Numpy matrix of float of size d x K
	def get_feature_matrix_from_covariance(self, C):
		'''Builds a feature matrix out of a covariance matrix @C as detailed in [Hoffman et al, 2014]'''
		assert is_of_type(C, "numpy.matrix")
		assert np.shape(C)[0] == np.shape(C)[1]
		assert np.linalg.det(C) > 0
		K = np.shape(C)[0]
		# Apply SVD
		V, D, _ = np.linalg.svd(C)
		assert all([i == K for i in np.shape(V)]) and len(D) == K
		idx = np.argwhere(np.array(D) > 0).flatten().tolist()
		N = len(idx)
		assert N > 0
		D = np.diag(D[idx])
		X = np.transpose(np.dot(V[:,idx], np.sqrt(D)))
		assert is_of_type(X, "numpy.matrix")
		return X, N, K

	#' @param A Numpy matrix: inverse of matrix
	#' @param x Numpy array
	#' @returns Aprime updates inverse of matrix with vector x using Sherman-Morrison formula 
	def iterative_inversion(self, A, x):
		return A - (np.dot(np.dot(A, x), np.dot(x.T, A)))/float(1+utils.matrix_norm(x, A)**2)

	def frank_wolfe(self, B, n=500):
		w = np.array([1]*self.K)
		V_inv = np.eye(self.N)
		for t in range(n):
			a_t = self.randf([float(np.max([self.X[:,b].T.dot(V_inv.dot(self.X[:,a])) for b in B])) for a in self.arms], np.max)
			V_inv = self.iterative_inversion(V_inv, self.X[:, a_t])
			e_a_t = np.array([float(int(a == a_t)) for a in self.arms])
			w = (t*w+e_a_t)/float(t+1)
		w = np.asarray(w, dtype=float).tolist()
		return w, np.max([float(utils.matrix_norm(self.X[:, b], V_inv)) for b in B])

	## Saddle-point Frank Wolfe in [Deguenne et al., 2020]
	## Solves AB = \min_{w \in Sigma_K} \max_{b \in B} ||b||_{(\sum_{a \in [K]} w_ax_ax_a^T)^{-1}}
	## by returning w, the optimal proportion vector which satisfies AB and the value of AB
	## Converges
	def saddle_point(self, B, n=500):
		w = np.array([1]*self.K)
		V_inv, V_t = np.eye(self.N), np.eye(self.N)
		for t in range(n):
			a_t = self.randf([float(utils.matrix_norm(self.X[:, a], V_inv.dot(V_t.dot(V_inv)))) for a in self.arms], np.max)
			b_t = B[self.randf([float(utils.matrix_norm(self.X[:, b], V_inv)) for b in B], np.max)]
			V_inv = self.iterative_inversion(V_inv, self.X[:, a_t])
			V_t += self.X[:, b_t].dot(self.X[:, b_t].T)
			e_a_t = np.array([float(int(a == a_t)) for a in self.arms])
			w = (t*w+e_a_t)/float(t+1)
		w = np.asarray(w, dtype=float).tolist()
		return w, np.max([float(utils.matrix_norm(self.X[:, b], V_inv)) for b in B])

	def update_empirical_means(self, candidates):
		for i in range(1, (self.K*self.T_init if (self.T_init > 0 and self.t <= self.K*self.T_init) else len(candidates))+1):
			arm = self.pulled_arms[-i]
			self.cum_sum[arm] += self.rewards[-i]
			self.means[arm] = self.cum_sum[arm]/float(self.na[arm])

	def update_linear_means(self, candidates):
		for i in range(1, (self.K*self.T_init if (self.T_init > 0 and self.t <= self.K*self.T_init) else len(candidates))+1):
			x = self.X[:, self.pulled_arms[-i]]
			self.B_inv = self.iterative_inversion(self.B_inv, x)
			self.b += self.rewards[-i]*np.array(x.tolist())[:,0]
			self.theta = np.dot(self.B_inv, self.b).reshape((1, self.N))
			self.means = np.array(np.dot(self.theta, self.X).tolist()[0])

	################################
	## ALGORITHMIC RULES          ##
	################################

	## RECOMMENDATION RULES

	## Default one
	def recommend_rule(self):
		idx = [int(id_) for id_ in utils.m_maximal(self.means.tolist(), self.m)]
		means = self.means[idx].tolist()
		assert all([m >= utils.m_max(self.means.tolist(), self.m) for m in means])
		return idx

	## SAMPLING RULES

	## Greedy sampling rule in LinGapE
	def pull_arm_greedy(self, A):
		assert not utils.is_of_type(self.best_arm, "NoneType")
		assert not utils.is_of_type(self.challenger, "NoneType")
		direction = self.X[:, self.best_arm]-self.X[:, self.challenger]
		uncertainty = [float(utils.matrix_norm(direction, self.iterative_inversion(A, self.X[:,i]))) for i in self.arms]
		a = self.arms[self.randf(uncertainty, np.min)]
		return [int(a)]

	## Optimized sampling rule in LinGapE
	def pull_arm_optimized(self, A=None):
		assert not utils.is_of_type(self.best_arm, "NoneType")
		assert not utils.is_of_type(self.challenger, "NoneType")
		p = self.ratio.get((self.best_arm, self.challenger))
		if (utils.is_of_type(p, "NoneType")):
			## https://math.stackexchange.com/questions/1639716/how-can-l-1-norm-minimization-with-linear-equality-constraints-basis-pu
			## what is called "method B"
			from scipy.optimize import linprog
			Aeq = np.concatenate((self.X, -self.X), axis=1)
			beq = (self.X[:,self.best_arm]-self.X[:,self.challenger])
			## coefficients of linear function to optimize
			F = np.ones((2*self.K, ))
			## nonnegative elements
			bounds = [(0, float("inf"))]*(2*self.K)
			solve = linprog(F, A_eq=Aeq, b_eq=beq, bounds=bounds)
			x = solve.x
			w = x[:self.K]-x[self.K:]
			assert solve.status == 0
			p = np.abs(w)
			p /= np.linalg.norm(w, 1)
			self.ratio.setdefault((self.best_arm, self.challenger), p)
		samplable_arms = [i for i in self.arms if (float(p[i]) > 0)]
		a = samplable_arms[self.randf([float(self.na[i]/float(p[i])) for i in samplable_arms], np.min)]
		return [int(a)]

	def reset_bandit_specific_parameters(self):
		raise NotImplemented

	def update(self, candidates):
		raise NotImplemented

	def sample(self):
		raise NotImplemented

	def stopping_rule(self):
		raise NotImplemented

#############################################################################################################################

####################################################
##  General index-based algorithm                 ##
####################################################

#' @param args Python dictionary
#' @param X NumPy matrix
#' @param m Python integer (0 < m < np.shape(X)[1])
#' @param problem custom GenericProblem instance (defined in "utils.py")
#' @param delta Python float (0 < delta < 1)
#' @param epsilon Python float (0 <= epsilon)
#' @param verbose Python bool
#' @param warning Python bool
#' @param params Python character string
#' @return bandit custom GIFA instance
class GIFA(ExploreMBandit):
	'''Subclass of ExploreMBandit. Returns the general skeleton of gap-index-based Top-m identification algorithms'''
	def __init__(self, args, X, m, problem, theta=None, X_type="feature", delta=0.05, epsilon=0., verbose=False, warning=False, params="", path_to_plots="../Results/", plot_step=100):
		args.setdefault("name", "GIFA")
		super(GIFA, self).__init__(args, X, m, problem, theta, X_type, delta, epsilon, verbose, warning, params, path_to_plots, plot_step)
		## Not initialized /!\
		self.index = Index(args)

	#Gap index
	def B_ij(self, i, j):
		return self.index.B_ij(i, j, self.t+int(self.T_init == 0), self.arg_index())

	#Variance on single arm (for sampling purposes)
	def var(self, i):
		return self.index.variance(i, self.t+int(self.T_init == 0), self.arg_index())

	#Confidence interval
	def CI(self, i, j=None):
		return self.index.CI(i, self.t+int(self.T_init == 0), self.arg_index(), j)

	def sample(self):
		self.J = self.compute_Jt()
		self.notJ = [a for a in self.arms if (a not in self.J)]
		self.indices = [self.compute_index(a) for a in self.arms]
		self.best_arm = self.compute_bt()
		indices_bt = [self.B_ij(a, self.best_arm) for a in self.notJ]
		self.challenger = self.notJ[self.randf(indices_bt, np.max)]
		return self.sampling_rule()

	## STOPPING RULES

	def tauLUCB(self):
		return float(self.B_ij(self.challenger, self.best_arm))

	def tauUGapE(self):
		return float(np.max([self.indices[i] for i in self.J]))

	## Note: this is theoretically valid only for m=1 and epsilon=0
	def tauChernoff(self, type_):
		if (self.na[self.best_arm] == 0):
			return self.epsilon-1
		## Chernoff stopping rule [Garivier and Kaufmann, 2016]
		self.kl_div = utils.kl_di(self.sigma)[type_]
		if (type_ == "gaussian"):
			Z = lambda c, b : self.na[c]*self.na[b]*(self.means[c]-self.means[b])**2/(self.na[c]+self.na[b])
		else:
			DZ = lambda c,b : (self.na[c]*self.means[c]+self.na[b]*self.means[b])/(self.na[c]+self.na[b])
			Z = lambda c,b : self.na[c]*self.kl_div(self.means[c], DZ(c, b))+self.na[b]*self.kl_div(self.means[b], DZ(c, b))
		return float(np.max([Z(c, self.best_arm) for c in self.arms if (c != self.best_arm and (self.na[c] > 0 or self.na[self.best_arm] > 0))]))

	def stopping_rule_chernoff(self, type_=["gaussian", "bernouilli"][0], verbose=True):
		if ((self.T_init == 0 and self.t == 1) or (self.T_init > 0 and self.t == self.T_init*self.K)):
			print("Warning! Using Chernoff stopping rule with m > 1 and epsilon > 0!")
		if (str(verbose) == "None"):
			verbose = self.verbose
		self.B = self.beta(self.delta)(self.t)-self.tauChernoff(type_)
		if (verbose and self.t%self.plot_step == 0 and self.plot_step <= self.t):
			print("B("+str(self.t)+") = "+str(self.B)+" ")
		return (self.B <= 0.)

	def stopping_rule(self, verbose=True):
		if (str(verbose) == "None"):
			verbose = self.verbose
		if (self.use_chernoff != "none"):
			return self.stopping_rule_chernoff(type_=self.use_chernoff, verbose=verbose)
		self.B = self.tau()
		if (verbose and self.t%self.plot_step == 0 and self.plot_step <= self.t):
			print("B("+str(self.t)+") = "+str(self.B)+" ")
		return (self.B <= self.epsilon)

	## RECOMMENDATION RULE

	def recommend_rule(self):
		idx = [int(id_) for id_ in self.J]
		return idx

	def reset_bandit_specific_parameters(self):
		raise NotImplemented

	## builds argument dictionary for index at time t > 0
	def arg_index(self):
		raise NotImplemented

	def compute_index(self, j):
		raise NotImplemented

	def compute_Jt(self):
		raise NotImplemented

	def compute_bt(self):
		raise NotImplemented

	def sampling_rule(self):
		raise NotImplemented

	def update(self, candidates):
		raise NotImplemented

####################################################
##  LUCB [Kalyanankrishnan et al, 2012]           ##
####################################################

#' @param args Python dictionary
#' @param X NumPy matrix
#' @param m Python integer (0 < m < np.shape(X)[1])
#' @param problem custom GenericProblem instance (defined in "utils.py")
#' @param delta Python float (0 < delta < 1)
#' @param epsilon Python float (0 <= epsilon)
#' @param verbose Python bool
#' @param warning Python bool
#' @param params Python character string
#' @return bandit custom LUCB instance
class LUCB(GIFA):
	'''Subclass of GIFA. Returns a LUCB bandit instance that solves the associated (@epsilon, @delta)-EXPLORE-@m problem with feature matrix @X, with arms defined in @problem'''
	def __init__(self, args, X, m, problem, theta=None, X_type="feature", delta=0.05, epsilon=0., verbose=False, warning=False, params="", path_to_plots="../Results/", plot_step=100):
		args.setdefault("name", "LUCB")
		super(LUCB, self).__init__(args, X, m, problem, theta, X_type, delta, epsilon, verbose, warning, params, path_to_plots, plot_step)
		self.index = DisjointNonContextualIndex({"beta": self.beta, "sigma": self.sigma, "KL_bounds": False, "problem": self.problem})
		if (self.is_greedy and not "_greedy" in self.plot_name):
			self.plot_name += "_greedy"
		if (self.use_chernoff != "none" and not "_chernoff" in self.plot_name):
			self.plot_name += "_chernoff="+self.use_chernoff

	def reset_bandit_specific_parameters(self):
		self.T_init = 1
		self.cum_sum = [0]*self.K
		self.confidence_intervals = np.zeros((self.K, 2))

	def arg_index(self):
		return {"na": self.na.tolist(), "means": self.means.tolist()}

	def compute_index(self, j):
		return self.B_ij(j, j)

	def compute_Jt(self):
		return utils.m_maximal(self.means.tolist(), self.m)

	def compute_bt(self):
		lower_bounds = self.confidence_intervals[self.J, 0].tolist()
		return self.J[self.randf(lower_bounds, np.min)]

	def sampling_rule(self):
		candidates = [self.best_arm, self.challenger]
		if (self.is_greedy):
			## Return arm with largest variance
			uncertainty = [float(self.var(a)) for a in candidates]
			return [candidates[self.randf(uncertainty, np.max)]]
		return candidates

	def update(self, candidates):
		self.update_empirical_means(candidates)
		for i in self.arms:
			self.confidence_intervals[i,:] = self.CI(i)

	def tau(self):
		return self.tauLUCB()

####################################################
##  KL-LUCB [Kaufmann et al., 2013]               ##
####################################################

#' @param args Python dictionary: must contain k1_diff:float > 0, alpha: ]1,+inf[, sigma > 0
#' @param X NumPy matrix
#' @param m Python integer (0 < m < np.shape(X)[1])
#' @param problem custom GenericProblem instance (defined in "utils.py")
#' @param delta Python float (0 < delta < 1)
#' @param epsilon Python float (0 <= epsilon)
#' @param verbose Python bool
#' @param warning Python bool
#' @param params Python character string
#' @return bandit custom KL-LUCB instance
class KLLUCB(LUCB):
	'''Subclass of LUCB. Returns a KL-LUCB bandit instance that solves the associated (@epsilon, @delta)-EXPLORE-@m problem with feature matrix @X, with arms defined in @problem, and bandit-specific arguments in @args: [@k1_diff, @alpha, @sigma]'''
	def __init__(self, args, X, m, problem, theta=None, X_type="feature", delta=0.05, epsilon=0., verbose=False, warning=False, params="", path_to_plots="../Results/", plot_step=100):
		assert all([x in list(args.keys()) for x in ["k1_diff", "alpha", "sigma"]])
		args.setdefault("name", "KL-LUCB")
		super(KLLUCB, self).__init__(args, X, m, problem, theta, X_type, delta, epsilon, verbose, warning, params, path_to_plots, plot_step)
		self.index = DisjointNonContextualIndex({"beta": self.beta, "sigma": self.sigma, "KL_bounds": True, "problem": self.problem})

#############################################################
##  Racing/Successive Elimination [Even-Dar et al., 2006]  ##
#############################################################

#' @param args Python dictionary: must contain sigma:float > 0, alpha: ]1,+inf[, eta: float > 0
#' @param X NumPy matrix
#' @param m Python integer (0 < m < np.shape(X)[1])
#' @param problem custom GenericProblem instance (defined in "utils.py")
#' @param delta Python float (0 < delta < 1)
#' @param epsilon Python float (0 <= epsilon)
#' @param verbose Python bool
#' @param warning Python bool
#' @param params Python character string
#' @return bandit custom Racing instance
class Racing(LUCB):
	'''Subclass of LUCB. Returns a Racing bandit instance that solves the associated (@epsilon, @delta)-EXPLORE-@m problem with feature matrix @X, with arms defined in @problem'''
	def __init__(self, args, X, m, problem, theta=None, X_type="feature", delta=0.05, epsilon=0., verbose=False, warning=False, params="", path_to_plots="../Results/", plot_step=100):
		args.setdefault("name", "Racing")
		super(Racing, self).__init__(args, X, m, problem, theta, X_type, delta, epsilon, verbose, warning, params, path_to_plots, plot_step)

	def reset_bandit_specific_parameters(self):
		self.T_init = 1
		super(Racing, self).reset_bandit_specific_parameters()
		## Remaining arms to sample
		self.R = [i for i in self.arms]
		## Selected & Discarded arms
		self.S, self.D = [], []

	def recommend_rule(self):
		if (len(self.S) != self.m):
			return (self.S+self.R+self.D)[:self.m]
		return self.S

	def stopping_rule(self, verbose=False):
		return (len(self.S) == self.m)

	def compute_Jt(self):
		return utils.m_maximal(list(map(float, self.means.tolist())), self.m-len(self.S))

	def sampling_rule(self):
		if (self.stopping_rule(verbose=False)):
			return []
		means_R = [float(self.means[i]) for i in self.R]
		arm_B, arm_W = self.R[self.randf(means_R, np.max)], self.R[self.randf(means_R, np.min)]
		index_B, index_W = self.B_ij(self.challenger, arm_B), self.B_ij(arm_W, self.best_arm)
		if (any([index < self.epsilon for index in [index_B, index_W]])):
			arm = [arm_B, arm_W][self.randf([index*int(index < self.epsilon) for index in [index_B, index_W]], np.max)]
			self.R = list(filter(lambda x : x != arm, self.R))
			if (arm == arm_B):
				self.S += [arm]
			else:
				self.D = [arm] + self.D
		return self.R

####################################################
##  KL-Racing [Kaufmann et al., 2013]             ##
####################################################

#' @param args Python dictionary: must contain sigma:float > 0, alpha: ]1,+inf[, eta: float > 0
#' @param X NumPy matrix
#' @param m Python integer (0 < m < np.shape(X)[1])
#' @param problem custom GenericProblem instance (defined in "utils.py")
#' @param delta Python float (0 < delta < 1)
#' @param epsilon Python float (0 <= epsilon)
#' @param verbose Python bool
#' @param warning Python bool
#' @param params Python character string
#' @return bandit custom KL-Racing instance
class KLRacing(Racing):
	'''Subclass of Racing. Returns a LUCB bandit instance that solves the associated (@epsilon, @delta)-EXPLORE-@m problem with feature matrix @X, with arms defined in @problem'''
	def __init__(self, args, X, m, problem, theta=None, X_type="feature", delta=0.05, epsilon=0., verbose=False, warning=False, params="", path_to_plots="../Results/", plot_step=100):
		assert all([x in list(args.keys()) for x in ["k1_diff", "alpha", "sigma"]])
		args.setdefault("name", "KL-Racing")
		super(KLRacing, self).__init__(args, X, m, problem, theta, X_type, delta, epsilon, verbose, warning, params, path_to_plots, plot_step)
		self.index = DisjointNonContextualIndex({"beta": self.beta, "sigma": self.sigma, "KL_bounds": True, "problem": self.problem})

####################################################
## UGapE [Gabillon et al., 2012]                  ##
####################################################

#' @param args Python dictionary
#' @param X NumPy matrix
#' @param m Python integer (0 < m < np.shape(X)[1])
#' @param problem custom GenericProblem instance (defined in "utils.py")
#' @param delta Python float (0 < delta < 1)
#' @param epsilon Python float (0 <= epsilon)
#' @param verbose Python bool
#' @param warning Python bool
#' @param params Python character string
#' @return bandit custom UGapE instance
class UGapE(GIFA):
	'''Subclass of GIFA. Returns a bandit instance that solves the associated (@epsilon, @delta)-EXPLORE-@m problem with feature matrix @X, with arms defined in @problem (UGapE [Gabillon et al., 2012])'''
	def __init__(self, args, X, m, problem, theta=None, X_type="feature", delta=0.05, epsilon=0., verbose=False, warning=False, params="", path_to_plots="../Results/", plot_step=100):
		args.setdefault("name", "UGapE")
		super(UGapE, self).__init__(args, X, m, problem, theta, X_type, delta, epsilon, verbose, warning, params, path_to_plots, plot_step)
		self.index = DisjointNonContextualIndex({"beta": self.beta, "sigma": self.sigma, "KL_bounds": False, "problem": self.problem})
		if (self.use_chernoff != "none" and not "_chernoff" in self.plot_name):
			self.plot_name += "_chernoff="+self.use_chernoff

	def reset_bandit_specific_parameters(self):
		self.T_init = 1
		self.cum_sum = [0]*self.K
		self.confidence_intervals = np.zeros((self.K, 2))

	def arg_index(self):
		return {"na": self.na.tolist(), "means": self.means.tolist()}

	def compute_index(self, j):
		B_ijs = [self.B_ij(a, j) for a in self.arms if (a != j)]
		index = utils.m_max(B_ijs, self.m)
		assert utils.is_of_type(index, "float")
		return index

	def compute_Jt(self):
		## here building J relies on arm indices
		minus_indices = [-self.compute_index(a) for a in self.arms]
		return utils.m_maximal(minus_indices, self.m)

	def compute_bt(self):
		lower_bounds = self.confidence_intervals[self.J, 0].tolist()
		return self.J[self.randf(lower_bounds, np.min)]

	def sampling_rule(self):
		## largest variance rule
		candidates = [self.best_arm, self.challenger]
		uncertainty = [self.var(a) for a in candidates]
		return [candidates[self.randf(uncertainty, np.max)]]

	def update(self, candidates):
		self.update_empirical_means(candidates)
		for i in self.arms:
			self.confidence_intervals[i,:] = self.CI(i)

	def tau(self):
		return self.tauUGapE()

####################################################
## LinUGapE                                       ##
####################################################

#' @param args Python dictionary
#' @param X NumPy matrix
#' @param m Python integer (0 < m < np.shape(X)[1])
#' @param problem custom GenericProblem instance (defined in "utils.py")
#' @param delta Python float (0 < delta < 1)
#' @param epsilon Python float (0 <= epsilon)
#' @param verbose Python bool
#' @param warning Python bool
#' @param params Python character string
#' @return bandit custom LinUGapE instance
class LinUGapE(UGapE):
	'''Subclass of UGapE. Returns a bandit instance that solves the associated (@epsilon, @delta)-EXPLORE-@m problem with feature matrix @X, with arms defined in @problem (UGapE [Gabillon et al., 2012] with paired contextual indices)'''
	def __init__(self, args, X, m, problem, theta=None, X_type="feature", delta=0.05, epsilon=0., verbose=False, warning=False, params="", path_to_plots="../Results/", plot_step=100):
		args.setdefault("name", "LinUGapE")
		super(LinUGapE, self).__init__(args, X, m, problem, theta, X_type, delta, epsilon, verbose, warning, params, path_to_plots, plot_step)
		self.index = PairedContextualIndex({"beta": self.beta, "X": self.X})
		if (self.use_chernoff != "none" and not "_chernoff" in self.plot_name):
			self.plot_name += "_chernoff="+self.use_chernoff

	def reset_bandit_specific_parameters(self):
		assert utils.is_of_type(self.sigma, "float") and self.sigma > 0
		assert utils.is_of_type(self.eta, "float") and self.eta > 0
		if (self.X_type == "covariance"):
			X, N, K = self.get_feature_matrix_from_covariance(X)
			self.X = X
			self.N = N
			self.K = K
		assert np.shape(self.X)[0] == self.N and np.shape(self.X)[1] == self.K and len(np.shape(self.X)) == 2
		self.T_init = 1
		self.lambda_ = self.sigma/float(self.eta)
		self.B_inv = 1/float(self.lambda_**2)*np.eye(self.N)
		self.b = np.zeros(self.N)
		self.theta = np.zeros((1, self.N))
		self.confidence_intervals = np.zeros((self.K, 2))

	def arg_index(self):
		return {"Sigma": (self.sigma**2)*self.B_inv, "theta": self.theta}

	def update(self, candidates):
		self.update_linear_means(candidates)
		for i in self.arms:
			self.confidence_intervals[i,:] = self.CI(i)

##########################################################################################
## m-LinGapE (adapted from [Xu et al., 2018], coincides with the true LinGapE for m=1   ##
##########################################################################################

#' @param args Python dictionary
#' @param X NumPy matrix
#' @param m Python integer (0 < m < np.shape(X)[1])
#' @param problem custom GenericProblem instance (defined in "utils.py")
#' @param delta Python float (0 < delta < 1)
#' @param epsilon Python float (0 <= epsilon)
#' @param verbose Python bool
#' @param warning Python bool
#' @param params Python character string
#' @return bandit custom (m)LinGapE instance
class LinGapE(GIFA):
	'''Subclass of GIFA. Returns a bandit instance that solves the associated (@epsilon, @delta)-EXPLORE-@m problem with feature matrix @X, with arms defined in @problem. Slightly modified from the Best Arm Identification version using the approach described as mLinGapE. Note: GLUCB (Zaki et al., 2019) == LinGapE with greedy arm sampling'''
	def __init__(self, args, X, m, problem, theta=None, X_type="feature", delta=0.05, epsilon=0., verbose=False, warning=False, params="", path_to_plots="../Results/", plot_step=100):
		args.setdefault("name", "m-LinGapE")
		super(LinGapE, self).__init__(args, X, m, problem, theta, X_type, delta, epsilon, verbose, warning, params, path_to_plots, plot_step)
		self.index = PairedContextualIndex({"beta": self.beta, "X": self.X})
		if (self.is_greedy and not "_greedy" in self.plot_name):
			self.plot_name += "_greedy"
		if (not self.is_greedy and not "_optimized" in self.plot_name):
			self.plot_name += "_optimized"
		if (self.use_chernoff != "none" and not "_chernoff" in self.plot_name):
			self.plot_name += "_chernoff="+self.use_chernoff

	def reset_bandit_specific_parameters(self):
		assert utils.is_of_type(self.sigma, "float") and self.sigma > 0
		assert utils.is_of_type(self.eta, "float") and self.eta > 0
		if (self.X_type == "covariance"):
			X, N, K = self.get_feature_matrix_from_covariance(X)
			self.X = X
			self.N = N
			self.K = K
		assert np.shape(self.X)[0] == self.N and np.shape(self.X)[1] == self.K and len(np.shape(self.X)) == 2
		self.T_init = 1
		self.hyper = 1.
		self.B_inv = 1/float(self.hyper**2)*np.eye(self.N)
		self.b = np.zeros(self.N)
		self.theta = np.zeros((1, self.N))

	def arg_index(self):
		return {"Sigma": (self.sigma**2)*self.B_inv, "theta": self.theta}

	def compute_index(self, j):
		index = float(np.max([self.B_ij(i, j) for i in self.notJ]))
		assert utils.is_of_type(index, "float")
		return index

	def compute_Jt(self):
		return utils.m_maximal(self.means.tolist(), self.m)

	def compute_bt(self):
		indices_J = [self.indices[i] for i in self.J]
		return self.J[self.randf(indices_J, np.max)]

	def sampling_rule(self):
		return (self.pull_arm_greedy if (self.is_greedy) else self.pull_arm_optimized)(self.B_inv)

	def update(self, candidates):
		self.update_linear_means(candidates)

	def tau(self):
		return self.tauLUCB()

class LinGapENoInit(LinGapE):
	'''Subclass of GIFA. Returns a bandit instance that solves the associated (@epsilon, @delta)-EXPLORE-@m problem with feature matrix @X, with arms defined in @problem. Slightly modified from the Best Arm Identification version using the approach described as mLinGapE. Note: GLUCB (Zaki et al., 2019) == LinGapE with greedy arm sampling'''
	def __init__(self, args, X, m, problem, theta=None, X_type="feature", delta=0.05, epsilon=0., verbose=False, warning=False, params="", path_to_plots="../Results/", plot_step=100):
		args.setdefault("name", "m-LinGapENoInit")
		super(LinGapENoInit, self).__init__(args, X, m, problem, theta, X_type, delta, epsilon, verbose, warning, params, path_to_plots, plot_step)
		self.index = PairedContextualIndex({"beta": self.beta, "X": self.X})
		if (self.is_greedy and not "_greedy" in self.plot_name):
			self.plot_name += "_greedy"
		if (not self.is_greedy and not "_optimized" in self.plot_name):
			self.plot_name += "_optimized"
		if (self.use_chernoff != "none" and not "_chernoff" in self.plot_name):
			self.plot_name += "_chernoff="+self.use_chernoff

	def reset_bandit_specific_parameters(self):
		super(LinGapENoInit, self).reset_bandit_specific_parameters()
		self.T_init = 0

####################################################
## LinLUCB                                      ##
####################################################

#' @param args Python dictionary: must contain sigma:float > 0, alpha: ]1,+inf[, eta: float > 0
#' @param X NumPy matrix
#' @param m Python integer (0 < m < np.shape(X)[1])
#' @param problem custom GenericProblem instance (defined in "utils.py")
#' @param delta Python float (0 < delta < 1)
#' @param epsilon Python float (0 <= epsilon)
#' @param verbose Python bool
#' @param warning Python bool
#' @param params Python character string
#' @return bandit custom LinLUCB instance
class LinLUCB(LUCB):
	'''Subclass of LUCB. Returns a LinLUCB bandit instance that solves the associated (@epsilon, @delta)-EXPLORE-@m problem with feature matrix @X, with arms defined in @problem, and bandit-specific arguments in @args: [@sigma, @alpha, @eta]'''
	def __init__(self, args, X, m, problem, theta=None, X_type="feature", delta=0.05, epsilon=0., verbose=False, warning=False, params="", path_to_plots="../Results/", plot_step=100):
		args.setdefault("name", "LinLUCB")
		super(LinLUCB, self).__init__(args, X, m, problem, theta, X_type, delta, epsilon, verbose, warning, params, path_to_plots, plot_step)
		self.index = DisjointContextualIndex({"beta": args["beta"], "X":X})
		if (self.is_greedy and not "_greedy" in self.plot_name):
			self.plot_name += "_greedy"
		if (self.use_chernoff != "none" and not "_chernoff" in self.plot_name):
			self.plot_name += "_chernoff="+self.use_chernoff

	def reset_bandit_specific_parameters(self):
		assert utils.is_of_type(self.alpha, "float") and self.alpha > 1
		assert utils.is_of_type(self.sigma, "float") and self.sigma > 0
		assert utils.is_of_type(self.eta, "float") and self.eta > 0
		if (self.X_type == "covariance"):
			X, N, K = self.get_feature_matrix_from_covariance(X)
			self.X = X
			self.N = N
			self.K = K
		assert np.shape(self.X)[0] == self.N and np.shape(self.X)[1] == self.K and len(np.shape(self.X)) == 2
		self.T_init = 0
		self.lambda_ = self.sigma/float(self.eta)
		assert self.lambda_ > 0
		self.B_inv = 1/float(self.lambda_**2)*np.eye(self.N)
		self.b = np.zeros(self.N)
		self.theta = np.zeros((1, self.N))
		self.confidence_intervals = np.zeros((self.K, 2))

	def arg_index(self):
		return {"Sigma": (self.sigma**2)*self.B_inv, "theta": self.theta}

	def update(self, candidates):
		self.update_linear_means(candidates)
		for i in self.arms:
			self.confidence_intervals[i,:] = self.CI(i)

	## Comment this function for default greedy version for LUCB
	## that is, UGapE-like sampling rule
	def sampling_rule(self):
		candidates = [self.best_arm, self.challenger]
		if (self.is_greedy):
			return self.pull_arm_greedy(self.B_inv)
		return candidates

####################################################
## LinGIFA                                        ##
####################################################

#' @param args Python dictionary
#' @param X NumPy matrix
#' @param m Python integer (0 < m < np.shape(X)[1])
#' @param problem custom GenericProblem instance (defined in "utils.py")
#' @param delta Python float (0 < delta < 1)
#' @param epsilon Python float (0 <= epsilon)
#' @param verbose Python bool
#' @param warning Python bool
#' @param params Python character string
#' @return bandit custom LinGIFA instance
class LinGIFA(UGapE):
	'''Subclass of UGapE. Returns a bandit instance that solves the associated (@epsilon, @delta)-EXPLORE-@m problem with feature matrix @X, with arms defined in @problem. More straightforward adaptation of UGapE [Gabillon et al., 2012] than LinGapE [Xu et al., 2018]'''
	def __init__(self, args, X, m, problem, theta=None, X_type="feature", delta=0.05, epsilon=0., verbose=False, warning=False, params="", path_to_plots="../Results/", plot_step=100):
		args.setdefault("name", "LinGIFA")
		super(LinGIFA, self).__init__(args, X, m, problem, theta, X_type, delta, epsilon, verbose, warning, params, path_to_plots, plot_step)
		self.index = PairedContextualIndex({"beta": self.beta, "X": self.X})
		if (self.is_greedy and not "_greedy" in self.plot_name):
			self.plot_name += "_greedy"
		if (self.use_chernoff != "none" and not "_chernoff" in self.plot_name):
			self.plot_name += "_chernoff="+self.use_chernoff

	def reset_bandit_specific_parameters(self):
		assert utils.is_of_type(self.alpha, "float") and self.alpha > 1
		assert utils.is_of_type(self.sigma, "float") and self.sigma > 0
		assert utils.is_of_type(self.eta, "float") and self.eta > 0
		if (self.X_type == "covariance"):
			X, N, K = self.get_feature_matrix_from_covariance(X)
			self.X = X
			self.N = N
			self.K = K
		assert np.shape(self.X)[0] == self.N and np.shape(self.X)[1] == self.K and len(np.shape(self.X)) == 2
		self.T_init = 0
		self.lambda_ = self.sigma/float(self.eta)
		assert self.lambda_ > 0
		self.B_inv = 1/float(self.lambda_**2)*np.eye(self.N)
		self.b = np.zeros(self.N)
		self.theta = np.zeros((1, self.N))
		self.confidence_intervals = np.zeros((self.K, 2))

	def arg_index(self):
		return {"Sigma": (self.sigma**2)*self.B_inv, "theta": self.theta}

	def compute_bt(self):
		indices_J = [self.indices[i] for i in self.J]
		return self.J[self.randf(indices_J, np.max)]

	def sampling_rule(self):
		candidates = [self.best_arm, self.challenger]
		if (self.is_greedy):
			return self.pull_arm_greedy(self.B_inv)
		else:
			return super(LinGIFA, self).sampling_rule()

	def update(self, candidates):
		self.update_linear_means(candidates)

class LinGIFAPlus(LinGIFA):
	'''Subclass of LinGIFA. Returns a bandit instance that solves the associated (@epsilon, @delta)-EXPLORE-@m problem with feature matrix @X, with arms defined in @problem. Only difference with LinGIFA is a less conservative stopping rule: B^{LinGIFA}(t) \leq B^{LinGIFAPlus}(t)'''
	def __init__(self, args, X, m, problem, theta=None, X_type="feature", delta=0.05, epsilon=0., verbose=False, warning=False, params="", path_to_plots="../Results/", plot_step=100):
		args.setdefault("name", "LinGIFAPlus")
		super(LinGIFAPlus, self).__init__(args, X, m, problem, theta, X_type, delta, epsilon, verbose, warning, params, path_to_plots, plot_step)

	def tau(self):
		return self.tauLUCB()

class LinIAA(LinGIFA):
	'''Subclass of LinGIFA. Returns a bandit instance that solves the associated (@epsilon, @delta)-EXPLORE-@m problem with feature matrix @X, with arms defined in @problem. More straightforward adaptation of UGapE [Gabillon et al., 2012] than LinGapE [Xu et al., 2018]'''
	def __init__(self, args, X, m, problem, theta=None, X_type="feature", delta=0.05, epsilon=0., verbose=False, warning=False, params="", path_to_plots="../Results/", plot_step=100):
		args.setdefault("name", "LinIAA")
		super(LinIAA, self).__init__(args, X, m, problem, theta, X_type, delta, epsilon, verbose, warning, params, path_to_plots, plot_step)
		self.index = DisjointContextualIndex({"beta": self.beta, "X": self.X})
		if (self.is_greedy and not "_greedy" in self.plot_name):
			self.plot_name += "_greedy"
		if (self.use_chernoff != "none" and not "_chernoff" in self.plot_name):
			self.plot_name += "_chernoff="+self.use_chernoff

class LinGIFAWithInit(LinGIFA):
	'''Subclass of LinGIFA. Returns a bandit instance that solves the associated (@epsilon, @delta)-EXPLORE-@m problem with feature matrix @X, with arms defined in @problem. More straightforward adaptation of UGapE [Gabillon et al., 2012] than LinGapE [Xu et al., 2018]'''
	def __init__(self, args, X, m, problem, theta=None, X_type="feature", delta=0.05, epsilon=0., verbose=False, warning=False, params="", path_to_plots="../Results/", plot_step=100):
		args.setdefault("name", "LinGIFAWithInit")
		super(LinGIFAWithInit, self).__init__(args, X, m, problem, theta, X_type, delta, epsilon, verbose, warning, params, path_to_plots, plot_step)
		self.index = DisjointContextualIndex({"beta": self.beta, "X": self.X})
		if (self.is_greedy and not "_greedy" in self.plot_name):
			self.plot_name += "_greedy"
		if (self.use_chernoff != "none" and not "_chernoff" in self.plot_name):
			self.plot_name += "_chernoff="+self.use_chernoff

	def reset_bandit_specific_parameters(self):
		super(LinGIFAWithInit, self).reset_bandit_specific_parameters()
		self.T_init = 1

#############################################################################################################################

####################################################
## Uniform sampling algorithms                    ##
####################################################

####################################################
## "True" Uniform sampling                        ##
####################################################

#' @param args Python dictionary: must contain T_init:int > 0
#' @param X NumPy matrix
#' @param m Python integer (0 < m < np.shape(X)[1])
#' @param problem custom GenericProblem instance (defined in "utils.py")
#' @param delta Python float (0 < delta < 1)
#' @param epsilon Python float (0 <= epsilon)
#' @param verbose Python bool
#' @param warning Python bool
#' @param params Python character string
#' @return bandit custom TrueUniform instance
class TrueUniform(ExploreMBandit):
	'''Subclass of ExploreMBandit. Returns a (truly) Uniform bandit instance that solves the associated (@epsilon, @delta)-EXPLORE-@m problem with feature matrix @X, with arms defined in @problem, and bandit-specific arguments in @args: [@T_init]'''
	def __init__(self, args, X, m, problem, theta=None, X_type="feature", delta=0.05, epsilon=0., verbose=False, warning=False, params="", path_to_plots="../Results/", plot_step=100):
		assert "T_init" in list(args.keys())
		assert utils.is_of_type(args["T_init"], "int") and args["T_init"] > 0
		args.setdefault("name", "TrueUniform")
		super(TrueUniform, self).__init__(args, X, m, problem, theta, X_type, delta, epsilon, verbose, warning, params, path_to_plots, plot_step)

	def reset_bandit_specific_parameters(self):
		self.B = float("inf")
		self.done = True

	def update(self, candidates):
		self.update_empirical_means(candidates)

#############################################################################################################################

####################################################
## Game-theoretic Algorithm-type Class            ##
####################################################

#' @param args Python dictionary
#' @param X NumPy matrix
#' @param m Python integer (0 < m < np.shape(X)[1])
#' @param problem custom GenericProblem instance (defined in "utils.py")
#' @param delta Python float (0 < delta < 1)
#' @param epsilon Python float (0 <= epsilon)
#' @param verbose Python bool
#' @param warning Python bool
#' @param params Python character string
#' @return bandit custom GenericGame instance
# [Deguenne et al., 2020]
class GenericGame(ExploreMBandit):
	'''Subclass of ExploreMBandit. Returns a bandit instance that solves the associated (@epsilon, @delta)-EXPLORE-@m problem with feature matrix @X, with arms defined in @problem, and bandit-specific arguments in @args: [@eta,@sigma].'''
	def __init__(self, args, X, m, problem, theta=None, X_type="feature", delta=0.05, epsilon=0., verbose=False, warning=False, params="", path_to_plots="../Results/", plot_step=100):
		assert epsilon == 0
		args.setdefault("name", "GenericGame")
		self.learner_type = AdaHedge
		super(GenericGame, self).__init__(args, X, m, problem, theta, X_type, delta, epsilon, verbose, warning, params, path_to_plots, plot_step)

	def reset_bandit_specific_parameters(self):
		assert utils.is_of_type(self.alpha, "float") and self.alpha > 1
		assert utils.is_of_type(self.sigma, "float") and self.sigma > 0
		assert utils.is_of_type(self.eta, "float") and self.eta > 0
		if (self.X_type == "covariance"):
			X, N, K = self.get_feature_matrix_from_covariance(X)
			self.X = X
			self.N = N
			self.K = K
		assert np.shape(self.X)[0] == self.N and np.shape(self.X)[1] == self.K and len(np.shape(self.X)) == 2
		from scipy.stats import gamma, norm, beta
		dis = [utils.prior_di, utils.prior_args_di(self.eta, self.K), utils.prior_pdfs, utils.prior_cdfs, utils.kl_di(self.sigma)]
		assert all([self.problem.type in list(di.keys()) for di in dis])
		self.prior = lambda prior_args : utils.prior_di[self.problem.type](prior_args[0,:], prior_args[1,:]).tolist()[0]
		self.prior_args = utils.prior_args_di(self.eta, self.K)[self.problem.type]
		self.prior_pdf = utils.prior_pdfs[self.problem.type]
		self.prior_cdf = utils.prior_cdfs[self.problem.type]
		## For convex game
		self.pa = np.matrix(np.ones((self.K, self.K)), dtype=int)
		## Chernoff stopping rule [Garivier and Kaufmann, 2016]
		self.kl_div = utils.kl_di(self.sigma)[self.problem.type]
		## Sample all arms once
		self.T_init = 1
		self.lambda_ = self.sigma/float(self.eta)
		assert self.lambda_ > 0
		self.B_inv = 1/float(self.lambda_**2)*np.eye(self.N)
		self.b = np.zeros(self.N)
		self.theta = np.zeros((1, self.N))
		self.cum_sum = [0]*self.K
		self.confidence_intervals = np.zeros((self.K, 2))
		## Threshold
		self.threshold = self.beta(1)

	## Player's turn
	def best_answer(self):
		i_star = utils.m_maximal(list(map(float, self.means.tolist())), self.m)
		assert len(i_star) == self.m
		return i_star

	def recommend_rule(self):
		return self.best_answer()

	def update(self, candidates):
		self.update_linear_means(candidates)
		self.threshold = self.beta(self.t)

	## Run the entire algorithm
	#' @param plot_rounds Python boolean
	#' @param verbatim Python boolean
	#' @returns result, nb result contains recommended set (ids+means), nb is the number of sampled arms
	def run(self, plot_rounds=True, verbatim=False):
		self.clear()
		self.initialization(plot_rounds, T_init=self.T_init)
		i = 0
		while (True):
			if (verbatim):
				print("** Round #"+str(i+1)),
			i += 1
			self.done = self.stopping_rule()
			if (self.done):
				break
			candidates = self.sample()
			self.learn_from_arms(candidates, plot_rounds)
			if (i > self.max_it):
				print("\n\nRun #"+str(self.current_n_simu)+" has not terminated (nrounds > "+str(i)+")\n\n")
				self.not_terminated += 1
				break
		if (plot_rounds):
			self.plot_current_round(self.sample())
		## FIRST element of result is always the recommended arm indices
		## other elements are lists with information on recommended arms
		result = self.recommend()
		return result, len(self.pulled_arms)

	def stopping_rule(self, verbose=True):
		if (str(verbose) == "None"):
			verbose = self.verbose
		tau, _  = self.GLRT()
		self.B = self.threshold-tau
		if (verbose and self.t%self.plot_step == 0 and self.plot_step <= self.t):
			print("B("+str(self.t)+") = "+str(self.B)+" ")
		return (0. >= self.threshold-tau)

	def sample(self):
		raise NotImplemented

	def GLRT(self):
		raise NotImplemented

####################################################
## LinGame [Degenne et al., 2020]                 ##
####################################################

#' @param args Python dictionary
#' @param X NumPy matrix
#' @param m Python integer (0 < m < np.shape(X)[1])
#' @param problem custom GenericProblem instance (defined in "utils.py")
#' @param delta Python float (0 < delta < 1)
#' @param epsilon Python float (0 <= epsilon)
#' @param verbose Python bool
#' @param warning Python bool
#' @param params Python character string
#' @return bandit custom LinGame instance
# [Deguenne et al., 2020]
class LinGame(GenericGame):
	'''Subclass of GenericGame. Returns a bandit instance that solves the associated (@epsilon, @delta)-Best Arm Identification problem with feature matrix @X, with arms defined in @problem, and bandit-specific arguments in @args: [@eta,@sigma].'''
	def __init__(self, args, X, m, problem, theta=None, X_type="feature", delta=0.05, epsilon=0., verbose=False, warning=False, params="", path_to_plots="../Results/", plot_step=100):
		assert m == 1
		## for function 'alt_min'
		assert problem.type == "gaussian"
		args.setdefault("name", "LinGame")
		super(LinGame, self).__init__(args, X, m, problem, theta, X_type, delta, epsilon, verbose, warning, params, path_to_plots, plot_step)
		if (not "_"+self.use_tracking+"-tracking" in self.plot_name and self.use_tracking != "D"):
			self.plot_name += "_"+self.use_tracking+"-tracking"

	def reset_bandit_specific_parameters(self):
		## For sampling rule: dictionary of learners by key=best arm
		self.learners = {}
		self.sum_w = np.zeros(self.K)
		super(LinGame, self).reset_bandit_specific_parameters()

	def alt_min(self, w, a, i_star):
		## best empirical arm using current estimate of means
		assert a != i_star
		sum_w = float(np.sum(w))
		w = w/sum_w
		direction = self.X[:,a]-self.X[:,i_star]
		sum_arms_matrix = np.zeros((self.N, self.N))
		for k in self.arms:
			sum_arms_matrix += w[k]*self.X[:,k].dot(self.X[:,k].T)
		Vinv = np.linalg.inv(sum_arms_matrix)
		denom = float(direction.T.dot(Vinv.dot(direction)))
		## closest point
		eta = float(sum_w*direction.T.dot(self.theta.T)/denom)
		lambda_ = self.theta - (eta/sum_w) * (Vinv.dot(direction)).T
		## divergence to that point
		val_ = float(0.5*sum_w*(direction.T.dot(self.theta.T))**2/denom)
		return val_, lambda_, i_star

	def GLRT(self, i_star=None):
		## transport using number of samplings for each arm
		if (str(i_star) == "None"):
			i_star = self.best_answer()[0]
		val = float("inf")
		other_arms = [a for a in self.arms if (a != i_star)]
		res_list = [self.alt_min(np.array([self.na[arm] for arm in self.arms]), a, i_star) for a in other_arms]
		alt = self.randf([r[0] for r in res_list], np.min)
		val, alternative = res_list[alt][0], res_list[alt][1]
		return val, (alt, alternative)

	def optimistic_gradient(self, lambda_):
		grads = np.zeros(self.K)
		for a in self.arms:
			ref_value = (self.theta-lambda_).dot(self.X[:,a])
			confidence_width = np.log(self.t)
			deviation = np.sqrt(2*confidence_width*float(utils.matrix_norm(self.X[:,a], self.B_inv)))
			if (ref_value > 0):
				grads[a] = 0.5*(ref_value+deviation)**2
			else:
				grads[a] = 0.5*(ref_value-deviation)**2
			grads[a] = min(grads[a], confidence_width)
		return grads

	def sample(self):
		i_star = self.best_answer()[0]
		learner = self.learners.get(i_star, None)
		if (str(learner) == "None"):
			learner = self.learner_type(self.K)
		## query the learner
		w = learner.act()
		## best response
		_, alts = self.GLRT(i_star=i_star)
		delta = self.optimistic_gradient(alts[1])
		learner.incur(-delta)
		## save learner
		di_learner = {}
		di_learner.setdefault(i_star, learner)
		self.learners.update(di_learner)
		## tracking
		na = np.array(self.na, dtype=float).tolist()
		if ("ForcedExploration" in self.use_tracking):
			undersampled = [int(na[a] <= np.sqrt(self.t)-0.5*self.K) for a in self.arms]
			if (sum(undersampled) > 0):
				w = np.array(undersampled)/float(sum(undersampled))
		if ("C" == self.use_tracking[-1]):
			## C-tracking
			self.sum_w += w
			sampled = self.randf([float(na[a]-self.sum_w[a]) for a in self.arms], np.min)
		elif ("D" == self.use_tracking[-1]):
			## D-tracking
			sampled = self.randf([float(na[a]-self.t*w[a]) for a in self.arms], np.min)
		else:
			raise ValueError("Tracking rule not implemented.")
		return [sampled]

########################################################################################################
########################################################################################################

####################################################
## Factory                                        ##
####################################################

#' @param bandit Python character string
#' @param args Python dictionary
#' @param X NumPy matrix of size d x K
#' @param m Python integer in [|1,K-1|]
#' @param problem GenericProblem instance
#' @param theta NumPy array
#' @param X_type Python character string
#' @param delta Python float in (0,1)
#' @param epsilon Python float >= 0
#' @param verbose Python bool
#' @param params Python character string
#' @param path_to_plots Python character string
#' @param plot_step Python int
#' @return beta lambda function of two integer arguments
def bandit_factory(bandit=None, args=None, X=None, m=None, problem=None, theta=None, X_type=None, delta=None, epsilon=None, verbose=None, warning=None, params=None, path_to_plots=None, plot_step=None):
	# Bonferroni correction
	delta_bai2explore = None if (utils.is_of_type(delta, "NoneType") or utils.is_of_type(delta, "NoneType")) else delta/float(m)
	di = {
		## Non contextual Top-m confidence-based algorithms
	     "LUCB": (lambda _ : LUCB),
	     "KL-LUCB": (lambda _ : KLLUCB),
	     "UGapE": (lambda _ : UGapE),
		## Non contextual Top-m elimination-based algorithms
	     "Racing": (lambda _ : Racing), 
	     "KL-Racing": (lambda _ : KLRacing),
		## Contextual Top-m confidence-based algorithms
	     "LinUGapE": (lambda _ : LinUGapE),
	     "LinLUCB": (lambda _ : LinLUCB),
	     "LinGapE": (lambda _ : LinGapE),
             "LinGapENoInit": (lambda _ : LinGapENoInit),
	     "LinGIFA": (lambda _ : LinGIFA),
	     "LinGIFAWithInit": (lambda _ : LinGIFAWithInit),
	     "LinGIFAPlus": (lambda _ : LinGIFAPlus),
	     "LinIAA": (lambda _ : LinIAA),
		## Non contextual Top-m uniform sampling algorithms
	     "TrueUniform": (lambda _ : TrueUniform),
		## (Non) contextual Top-m/Top-1 Thompson sampling algorithms
	     "LinGame": (lambda _ : LinGame),
	}
	'''Factory for bandits: returns a ExploreMBandit instance'''
	if (utils.is_of_type(bandit, "NoneType")):
		return list(di.keys())
	assert utils.is_of_type(args, "dict")
	assert utils.is_of_type(bandit, "str")
	if (not (bandit in list(di.keys()))):
			print("\""+bandit+"\" not in "+str(list(di.keys())))
			raise ValueError
	return di[bandit](0)(args, X, m, problem, theta, X_type, delta, epsilon, verbose, warning, params, path_to_plots, plot_step)
