# coding: utf-8

import subprocess as sb
import numpy as np
import pandas as pd
import os

import utils

####################################################
## DATA GENERATION/LOADING                        ##
####################################################

data_list = ["linear", "logistic", "classic"]
simulated_list = {
	"linear": (lambda x, theta : np.dot(np.transpose(x), theta)), 
	"logistic": (lambda x, theta : 1/(1+np.exp(-np.dot(x.T, theta)))),
}

## build ith canonical vector from R^b
def e(i, b):
	assert i <= b
	e_i = [0]*b
	e_i[i-1] = 1
	return np.matrix([e_i]).T

def import_df(fname):
	df = pd.read_csv(fname)
	df.index = list(map(str, df[df.columns[0]]))
	df = df.loc[~df.index.duplicated()]
	df = df.drop(columns=[df.columns[0]])
	return df

def same_order_df(ids, df_list, axis_list):
	assert len(df_list) == len(axis_list)
	ids_ = list(filter(lambda idx : all([idx in df.columns for df in df_list]), ids))
	assert len(ids_) > 0
	df_res_list = [None]*len(df_list)
	for sdf, df in enumerate(df_list):
		assert axis_list[sdf] in range(2)
		if (axis_list[sdf] == 0):
			df_res_list[sdf] = df.loc[ids_]
		else:
			df_res_list[sdf] = df[ids_]
	return df_res_list

#' @param args Python dictionary of strings
#' @param folder_path Python character string
#' @param normalized Python bool
def create_scores_features(args, folder_path, normalized=False):
	'''Compute/retrieve feature matrix + normalized, if @normalized set to True, "oracle" scores from DR data @data'''
	data = args.data
	assert utils.is_of_type(data, "str")
	assert utils.is_of_type(normalized, "bool")
	#################################
	## Using drug repurposing data ##
	#################################
	if (not (data in data_list)):
		assert not (utils.is_of_type(data, "NoneType"))
		assert data == "epilepsy"
		from constants import dr_folder
		## Not considering the toy DR problem with 10 arms in the paper
		if (args.small_K != 10):
			## Arm features
			X = import_df(dr_folder+data+"_signatures_nonbinarized.csv")
			## Signatures that will be used to compute phenotypes through GRN
			## S := binarize(X)
			S = import_df(dr_folder+data+"_signatures_binarized.csv")
			## "True" drug scores
			A = import_df(dr_folder+data+"_scores.csv")
			## Ordered by drug signature ids
			A, X, S = same_order_df(list(X.columns), [A, X, S], [0, 1, 1])
			names = list(A["drug_name"])
			scores = list(map(float, A["score"]))
			df_di = {"S": S, "X": X, "names": names}
			X = np.matrix(X.values)
		## Subset of drugs where rewards were pre-recorded
		else:
			file_="rewards_cosine_10drugs_18samples"
			file_features="epilepsy_signatures.csv"
			## Known anti-epileptics
			names = ["Hydroxyzine", "Acetazolamide", "Pentobarbital", "Topiramate", "Diazepam"]
			## Known pro-convulsants
			names += ["Dmcm", "Brucine", "Fipronil", "Flumazenil", "Fg-7142"]
			assert len(names) == 10
			drug_ids, drug_positions = utils.get_drug_id(names, dr_folder+file_+".txt")
			assert not any([str(s) == "None" for s in drug_ids])
			A = import_df(dr_folder+data+"_scores.csv")
			drug_cids = A.index
			A.index = A["drug_name"]
			A["drug_cid"] = drug_cids
			drug_cids = list(map(str, A.loc[names]["drug_cid"]))
			assert len(drug_cids) == len(names)
			X = import_df(dr_folder+data+"_signatures.csv")
			S = import_df(dr_folder+data+"_signatures_binarized.csv")
			## Ordered by drug signature ids
			X, S = same_order_df(drug_cids, [X, S], [1]*2)
			rewards = pd.read_csv(dr_folder+file_+".csv", sep=" ", header=None)
			means = rewards.mean(axis=0).values
			scores = [float(means[i]) for i in drug_positions]
			df_di = {"S": S, "X": X, "names": names}
			X = np.matrix(X.values)
	#################################
	## "Classic" linear bandit     ##
	#################################
	elif (data == "classic"):
		assert utils.is_of_type(args.omega, "float")
		print("Omega = " + str(round(args.omega, 3)))
		assert args.small_K and args.m and args.omega
		if (args.problem == "bernouilli"):
			assert np.cos(args.omega) >= 0
		## canonical base in R^(K-1), modification from case m=1
		## arms 1, ..., m have rewards == 1
		## arm m+1 has reward cos(omega)
		## arm m+2, ..., K have rewards == 0
		m, K, N, omega = args.m, args.small_K, args.small_K-1, args.omega
		assert m < N
		X = np.matrix(np.eye(N, K))
		X[0,:m] = 1
		X[:,(m+1):] = X[:,m:(K-1)]
		X[:,m] = np.cos(omega)*e(1, N)+np.sin(omega)*e(m+1, N)
		theta = e(1, N)
		scores = simulated_list["linear"](X, theta).flatten().tolist()[0]
	#################################
	## Using simulated data        ##
	#################################
	## same setting than the one where complexity constants are compared
	elif (data in list(simulated_list.keys())):
		max_it_gen = 500
		assert args.small_K
		assert args.small_N
		N, K = args.small_N, args.small_K
		matrix_file = folder_path+"generated_matrix_N="+str(N)+"_K="+str(K)+".csv"
		if (not os.path.exists(matrix_file)):
			done = False
			it = 0
			while (not done and it < max_it_gen):
				## Normalizing the feature matrix
				X = np.matrix(np.random.normal(0, args.vr, (N, K)))
				X /= np.linalg.norm(X, 2)
				done = (np.linalg.matrix_rank(X) >= K)
				it += 1
			if (it == max_it_gen):
				print("Det value: "+str(np.linalg.det(np.dot(X.T, X))))
				print("Got unlucky...")
			np.savetxt(matrix_file, X)
		else:
			X = np.matrix(np.loadtxt(matrix_file), dtype=float)
		theta = e(1, N)
		scores = simulated_list[data](X, theta).flatten().tolist()[0]
	else:
		print("Data type not found!")
		raise ValueError
	if (not data in list(simulated_list.keys())):
		## Linear regression to find the "true" theta
		theta = np.linalg.inv(X.dot(X.T)).dot(X.dot(np.array(scores).T).T)
		## residual np.linalg.norm(X.T.dot(theta)-scores, 2)
		theta_file = folder_path+data+"_theta_K="+str(args.small_K)+".csv"
		np.savetxt(theta_file, theta)
	if (data in data_list):
		names = None
		df_di = {}
	assert theta.size == np.shape(X)[0]
	if (data in list(simulated_list.keys()) or data in ["classic"]):
		assert len(scores) == args.small_K 
	## If Bernouilli arms: means must belong to [0,1]
	if (args.problem == "bernouilli" and data in list(simulated_list.keys()) and data != "classic"):
		X = np.matrix(np.random.normal(0.5, 0.5, (args.small_N, args.small_K)))
		X /= np.linalg.norm(X, 2)
		theta = np.matrix(np.random.normal(0.5, 0.5, (args.small_N, 1)))
		theta /= np.linalg.norm(theta, 2)
		scores = list(map(float, theta.T.dot(X).tolist()[0]))
		assert np.all(np.array(scores) >= 0) and np.all(np.array(scores) <= 1)
	if (data == "linear"):
		## Print Boolean test on complexity constants
		from compare_complexity_constants import compute_H_UGapE, compute_H_optimized_LinGapE
		H_LinGapE = compute_H_optimized_LinGapE(theta, X, args.epsilon, args.m)
		H_UGapE = compute_H_UGapE(theta, X, args.epsilon, args.m)
		print("Is H_LinGapE < 2*H_UGapE? : "+str(H_LinGapE < 2*H_UGapE))
		with open(folder_path+data+"_boolean_test_UGapE_LinGapE_"+data+"N="+str(N)+"_K="+str(K)+"_m="+str(args.m)+".txt", "w+") as f:
			s_ = ["H_LinGapE = "+str(H_LinGapE)]
			s_.append("H_UGapE = "+str(H_UGapE))
			s_.append("Is H_LinGapE < 2*H_UGapE? : "+str(H_LinGapE < 2*H_UGapE))
			f.write("\n".join(s_))
	return X, scores, theta, names, df_di
