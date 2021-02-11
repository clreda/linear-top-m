# coding: utf-8

import numpy as np

from utils import is_of_type, is_of_type_LIST, is_of_type_OPTION

####################################################
## Exploration rate types                         ##
####################################################

## For confidence-interval-based algorithms

def AlphaDependentBeta(alpha, delta, X):
	assert is_of_type(alpha, "float")
	assert alpha > 1
	assert is_of_type(X, "numpy.matrix")
	assert is_of_type(delta, "float")
	assert 0 < delta and delta < 1
	from scipy.special import zeta
	z_alpha = zeta(alpha)
	K = np.shape(X)[1]
	return (lambda t : np.log(K*z_alpha*(t**alpha)/delta))

# [Kalyanakrishnan et al., 2012]
def LUCB1Beta(delta, X):
	assert is_of_type(X, "numpy.matrix")
	assert is_of_type(delta, "float")
	assert 0 < delta and delta < 1
	K = np.shape(X)[1]
	return (lambda t : np.log(5*K*(t**4)/(4*delta)))

def HoeffdingBeta(delta, sigma):
	assert is_of_type(sigma, "float")
	assert is_of_type(delta, "float")
	assert 0 < delta and delta < 1
	assert sigma > 0
	return (lambda t : np.log(1/float(delta)))

def HeuristicBeta(delta):
	assert is_of_type(delta, "float")
	assert 0 < delta and delta < 1
	return (lambda t : np.log((np.log(t)+1)/float(delta)))

def HeuristicLinearBeta(delta, X):
	assert is_of_type(X, "numpy.matrix")
	assert is_of_type(delta, "float")
	assert 0 < delta and delta < 1
	N, K = np.shape(X)
	return (lambda t : np.log(1/float(delta))+N*np.log(t)/float(2))

def SuperHeuristicLinearBeta(delta, X):
	assert is_of_type(X, "numpy.matrix")
	assert is_of_type(delta, "float")
	assert 0 < delta and delta < 1
	N, K = np.shape(X)
	return (lambda t : np.log(1/float(delta))+N*np.log(np.log(t+1))/float(2))

# [Kaufmann et al., 2013]
# [Kaufmann, 2014]
def KLLUCBBeta(alpha, k1_diff, X, delta):
	assert is_of_type(alpha, "float")
	assert alpha > 1
	assert is_of_type(k1_diff, "float")
	assert k1_diff > 0
	assert is_of_type(X, "numpy.matrix")
	assert is_of_type(delta, "float")
	assert 0 < delta and delta < 1
	K = np.shape(X)[1]
	k1 = 1+1/float(alpha-1)+k1_diff
	B = lambda t : k1*K*(t**alpha)/delta
	return (lambda t : np.log(k1*K*(t**alpha)/delta))

# [Abbasi-Yadkhori et al., 2011]
def FrequentistBeta(X, sigma, eta, S, delta):
	assert is_of_type(sigma, "float")
	assert sigma > 0
	assert is_of_type(eta, "float")
	assert eta > 0
	assert is_of_type(X, "numpy.matrix")
	assert is_of_type(S, "float")
	assert S > 0
	assert is_of_type(delta, "float")
	assert 0 < delta and delta < 1
	N, K = np.shape(X)
	L = np.max([np.linalg.norm(X[:,i], 2) for i in range(K)])
	lambda_ = float(sigma/float(eta))
	frequentist = lambda t : np.sqrt(2*np.log(1/float(delta))+N*np.log(1+(t+1)*L**2/float(lambda_**2*N)))+lambda_/float(sigma)*S
	## In linear bandits, C = sqrt{2*beta}
	return (lambda t : 0.5*(frequentist(t))**2)

# [Garivier et al., 2016]
def InformationalBeta(X, delta):
	assert is_of_type(X, "numpy.matrix")
	assert is_of_type(delta, "float")
	assert 0 < delta and delta < 1
	N, K = np.shape(X)
	return (lambda t : np.log(2*t*(K-1))/float(delta))

# [Garivier et al., 2016]
def DeviationalBeta(alpha, X, delta):
	assert is_of_type(alpha, "float")
	assert alpha > 1
	assert is_of_type(X, "numpy.matrix")
	assert is_of_type(delta, "float")
	assert 0 < delta and delta < 1
	N, K = np.shape(X)
	from scipy.special import zeta
	z_alpha = zeta(alpha)
	C = (K-1)*z_alpha
	return (lambda t : np.log(C*(t**alpha)/float(delta)))

####################################################
## Factory                                        ##
####################################################

#' @param beta Python character string
#' @param args Python dictionary
#' @return beta lambda function of two integer arguments
def beta_factory(beta, args):
	'''Factory for betas: returns a lambda function for the exploration rate initialized on problem values stored in @args'''
	assert is_of_type(args, "dict")
	assert is_of_type(beta, "str")
	di = {
		"AlphaDependent": (lambda _ : AlphaDependentBeta(args["alpha"], args["delta"], args["X"])),
		"LUCB1": (lambda _ : LUCB1Beta(args["delta"], args["X"])), 
		"Hoeffding": (lambda _ : HoeffdingBeta(args["delta"], args["sigma"])), 
		"Heuristic": (lambda _ : HeuristicBeta(args["delta"])), 
		"HeuristicLinear": (lambda _ : HeuristicLinearBeta(args["delta"], args["X"])), 
		"SuperHeuristicLinear": (lambda _ : SuperHeuristicLinearBeta(args["delta"], args["X"])),
		"KLLUCB": (lambda _ : KLLUCBBeta(args["alpha"], args["k1_diff"], args["X"], args["delta"])),
		"Frequentist": (lambda _ : FrequentistBeta(args["X"], args["sigma"], args["eta"], args["S"], args["delta"])),    
		"Informational": (lambda _ : InformationalBeta(args["X"], args["delta"])), 
		"Deviational": (lambda _ : DeviationalBeta(args["alpha"], args["X"], args["delta"])), 
	}
	if (not (beta in list(di.keys()))):
		print("\""+beta+"\" not in "+str(list(di.keys())))
		raise ValueError
	return di[beta](0)
