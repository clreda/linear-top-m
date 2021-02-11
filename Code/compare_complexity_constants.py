#coding: utf-8

## Command
# ```python compare_complexity_constants.py```

import os
import pickle
import subprocess as sb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import seed
from scipy.optimize import linprog

seed(123456789)
epsilon=0
sigma=1.

###############
## UTILS     ##
###############

#' @param theta Numpy array size (N,1)
#' @param X Numpy matrix size (N,K)
#' @param epsilon Python float
#' @param m Python integer 0 < m < K
#' @return mu, best_set, delta: 
#               - means (in increasing order of arm id), 
#               - best set (best arm ids), 
#               - arm gaps (in increasing order of arm id)
def compute_constants(theta, X, epsilon, m):
	N, K = np.shape(X)
	mu = X.T.dot(theta).flatten().tolist()[0]
	best_set = np.argsort(mu)[-m:].tolist()
	m1_arm, m_arm = np.argsort(mu)[-m-1:-m+1].tolist()
	delta = [mu[k]-mu[m1_arm] if (k in best_set) else mu[m_arm]-mu[k] for k in range(K)]
	return mu, best_set, delta

## https://math.stackexchange.com/questions/1639716/how-can-l-1-norm-minimization-with-linear-equality-constraints-basis-pu
#' @param X Numpy matrix size (N,K)
#' @param i arm id
#' @param j arm id i != j
#' @param w_star_di Python dictionary keys: integer pairs, values: Python list
#' @param verbose Python boolean
#' @return w Python list
def compute_w_star(X, i, j, w_star_di, verbose=False):
	w = w_star_di.get((i,j), None)
	K_ = np.shape(X)[1]
	if (str(w) == "None"):
		Aeq = np.concatenate((X, -X), axis=1)
		beq = X[:,i]-X[:,j]
		F = np.ones((2*K_, ))
		bounds = [(0, float("inf"))]*(2*K_)
		x = linprog(F, A_eq=Aeq, b_eq=beq, bounds=bounds).x
		if ("float" in str(type(x))):
			raise ValueError("solution x = "+str(x))
		w = list(map(float, (x[:K_]-x[K_:]).tolist()))
		if (verbose):
			print("\tw*(i,j) = "+str(w)+" (i="+str(i)+",j="+str(j)+")")
		w_star_di.setdefault((i,j), w)
	return w

#' @param theta Numpy array size (N,1)
#' @param X Numpy matrix size (N,K)
#' @param epsilon Python float
#' @param m Python integer 0 < m < K
#' @param verbose Python boolean
#' @return H Python float
def compute_H_optimized_LinGapE(theta, X, epsilon, m, verbose=False):
	N, K = np.shape(X)
	mu, best_set, delta = compute_constants(theta, X, epsilon, m)
	w_star_di = {}
	H = 0
	for a in range(K):
		arm_constant = -1.0
		for i in range(K):
			for j in range(i+1, K):
				denom = max(epsilon, 1/float(3)*(epsilon+max(delta[i], delta[j])))**2
				w_ij = compute_w_star(X, i, j, w_star_di, verbose=verbose)
				wa_ij = abs(w_ij[a])/float(denom)
				if (arm_constant < wa_ij):
					arm_constant = wa_ij
		assert arm_constant > -1
		H += arm_constant
	return H

#' @param theta Numpy array size (N,1)
#' @param X Numpy matrix size (N,K)
#' @param epsilon Python float
#' @param m Python integer 0 < m < K
#' @param verbose Python boolean
#' @return H Python float
def compute_H_UGapE(theta, X, epsilon, m):
	N, K = np.shape(X)
	mu, best_set, delta = compute_constants(theta, X, epsilon, m)
	H = sum([1./float(max(epsilon, 1/2.*(epsilon+delta[a]))**2) for a in range(K)])
	return H

####################
## EXPERIMENT     ##
####################

if (__name__ == "__main__"):

	## Count number of randomly generated instances where H(LinGapE) < H(UGapE)
	result_folder = "compare_complexity_constants_results/"
	if (not os.path.exists(result_folder)):
		sb.call("mkdir "+result_folder, shell=True)
	niter = 1000
	nb_arms_range = range(10, 50+1, 10)
	nb_dims_range = [0.5]+range(1, 2+1)
	var_range = [0.25, 0.5]
	for K in nb_arms_range:
		for N_perc in nb_dims_range:
			N = int(N_perc*K)
			theta = np.matrix([1.]+[0.]*(N-1)).reshape((N,1))
			for vr in var_range:
				save_fname = result_folder+"percentage_generated_instances_K="+str(K)+"_N="+str(N)+"_var="+str(vr)+".pck"
				if (not os.path.exists(save_fname)):
					print("\n* "+save_fname)
					m=int(K/3.)+1
					iter_=total=0
					print("Iteration\tH(LinGapE) < 2H(UGapE)\tH(LinGapE)\tH(UGapE)")
					while (iter_ < niter):
						X = np.matrix(np.random.normal(0, vr, size=N*K)).reshape((N,K))
						X /= np.linalg.norm(X, 2)
						try:
							H_LinGapE = compute_H_optimized_LinGapE(theta, X, epsilon, m)
						except:
							continue
						H_UGapE = compute_H_UGapE(theta, X, epsilon, m)
						boolean = int(H_LinGapE < 2./float(sigma**2)*H_UGapE)
						total += boolean
						print("\t".join(list(map(str, [iter_+1, boolean, int(H_LinGapE), int(H_UGapE)]))))
						iter_ += 1
					res = {"total": total, "niter":niter, "theta": theta.T.tolist(), "K": K, "N": N, "m": m, "D": vr}
					with open(save_fname, "w+") as f:
						pickle.dump(res, f)
				else:
					with open(save_fname, "r") as f:
						res = pickle.load(f)
				for key in list(res.keys()):
					if (key != "theta"):
						print(key+" = "+str(res[key]))
				print("Percentage of LinGapE < UGapE: "+str(round(res["total"]*100/float(res["niter"]), 2))+"% (niter="+str(res["niter"])+")")
