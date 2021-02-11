# coding: utf-8

import numpy as np

import utils

####################################################
## Indices                                        ##
####################################################

#' @param args Python dictionary
#' index B(i,j)(t) >= mu_i-mu_j at time t
class Index(object):
	def __init__(self, args):
		assert all([x in list(args.keys()) for x in ["beta"]])
		args.setdefault("name", "Index")
		for attr in list(args.keys()):
			setattr(self, attr, args[attr])
		self.t = 0
		self.B_di = {}	

	def B_ij(self, i, j, t, args=None):
		if (t != self.t):
			self.B_di = {}
			self.t = t
		b = self.B_di.setdefault((i,j), float(self.index(i,j,t, args=args)))
		return b

	def CI(self, i, t, args=None, j=None):
		## "Individual" confidence interval on mu_i
		if (str(j) == "None"):
			return self.confidence_interval(i, t, args=args)
		## Gap confidence interval on mu_i-mu_j
		else:
			return [-self.B_ij(j, i, t, args), self.B_ij(i, j, t, args)]

	def index(self, i, j, t, args=None):
		raise NotImplemented

	def gap(self, i, args=None, j=None):
		raise NotImplemented

	def variance(self, i, t, args=None, j=None):
		raise NotImplemented

	def confidence_interval(self, i, t, args=None):
		raise NotImplemented

#' @param args Python dictionary
class ContextualIndex(Index):
	def __init__(self, args):
		assert all([x in list(args.keys()) for x in ["beta", "X"]])
		args.setdefault("name", "ContextualIndex")
		super(ContextualIndex, self).__init__(args)

	def gap(self, i, args, j=None):
		keys = ["theta"]
		assert all([k in list(args.keys()) for k in keys])
		assert all([str(x) != "None" for x in keys])
		assert np.shape(args["theta"])[1] == np.shape(self.X)[0]
		assert all([0 <= x and x < np.shape(self.X)[1] for x in [i]])
		if (str(j) != "None"):
			assert all([0 <= x and x < np.shape(self.X)[1] for x in [j]])
			gap_ = np.dot(args["theta"], self.X[:, i]-self.X[:,j])
		else:
			gap_ = np.dot(args["theta"], self.X[:, i])
		return gap_

	def variance(self, i, t, args, j=None):
		keys = ["Sigma"]
		assert all([k in list(args.keys()) for k in keys])
		assert all([str(x) != "None" for x in keys])
		assert 0 <= i and i < np.shape(self.X)[1]
		assert all([x == np.shape(self.X)[0] for x in np.shape(args["Sigma"])])
		assert t > 0
		v = lambda x : float(utils.matrix_norm(x, args["Sigma"])*np.sqrt(2*self.beta(t)))
		## individual
		if (str(j) == "None"):
			return v(self.X[:, i])
		## paired
		else:
			assert 0 <= j and j < np.shape(self.X)[1]
			return v(self.X[:, i]-self.X[:, j])

#' @param args Python dictionary
class NonContextualIndex(Index):
	def __init__(self, args):
		assert all([x in list(args.keys()) for x in ["beta", "sigma", "KL_bounds", "problem"]])
		args.setdefault("name", "NonContextualIndex"+("_KLbounds" if (args["KL_bounds"]) else ""))
		super(NonContextualIndex, self).__init__(args)
		assert self.problem.type in list(utils.kl_di_bounds(self.sigma).keys())
		self.kl_div = utils.kl_di_bounds(self.sigma)[self.problem.type]

	def gap(self, i, args, j=None):
		keys = ["means"]
		assert all([k in list(args.keys()) for k in keys])
		assert all([str(x) != "None" for x in keys])
		assert i < len(args["means"])
		gap_ = args["means"][i]
		if (str(j) != "None"):
			assert j < len(args["means"])
			gap_ -= args["means"][j]
		return gap_

	def variance(self, i, t, args, j=None, minv=1e-10):
		keys = ["na"]
		assert all([k in list(args.keys()) for k in keys])
		assert all([str(x) != "None" for x in keys])
		assert all([x >= 0 for x in args["na"]])
		assert i < len(args["na"])
		assert t > 0
		if (self.KL_bounds):
			v = lambda a : self.beta(t)/float(max(args["na"][a], minv))
		else:
			v = lambda a : float(np.sqrt(2*self.sigma**2*self.beta(t)/float(max(args["na"][a], minv))))
		## individual
		if (str(j) == "None"):
			return v(i)
		## paired
		else:
			assert j < len(args["na"])
			return v(i)+v(j)

	def KL(self, i, t, args, which):
		assert utils.is_of_type(which, "str") and which in ["upper", "lower"]
		mean_, var_ = self.gap(i, args), self.variance(i, t, args)
		lower, upper = [mean_,1.] if (which=="upper") else [0.,mean_]
		return self.kl_div(mean_, var_, lower, upper)

#######################
## Fully implemented ##
#######################

#' @param args Python dictionary
class DisjointContextualIndex(ContextualIndex):
	def __init__(self, args):
		args.setdefault("name", "DisjointContextualIndex")
		super(DisjointContextualIndex, self).__init__(args)

	def index(self, i, j, t, args):
		return self.gap(i, args, j=j)+self.variance(i, t, args)+self.variance(j, t, args)

	def confidence_interval(self, i, t, args):
		mean_ = self.gap(i, args)
		var_ = self.variance(i, t, args)
		return [mean_ - var_, mean_ + var_] 

#' @param args Python dictionary
class DisjointNonContextualIndex(NonContextualIndex):
	def __init__(self, args):
		assert "KL_bounds" in list(args.keys())
		args.setdefault("name", "DisjointNonContextualIndex"+("_KLbounds" if (args["KL_bounds"]) else ""))
		super(DisjointNonContextualIndex, self).__init__(args)

	def index(self, i, j, t, args):
		if (self.KL_bounds):
			u_i, l_j = self.KL(i, t, args, which="upper"), self.KL(j, t, args, which="lower")
			return u_i-l_j
		else:
			gap_, var_ = self.gap(i, args, j=j), self.variance(i, t, args, j=j)
			return gap_+var_

	def confidence_interval(self, i, t, args):
		if (self.KL_bounds):
			u_i, l_i = [self.KL(i, t, args, which=wh) for wh in ["upper", "lower"]]
			return [l_i, u_i]
		else:
			mean_, var_ = self.gap(i, args), self.variance(i, t, args)
			return [mean_-var_, mean_+var_]

#' @param args Python dictionary
class PairedContextualIndex(ContextualIndex):
	def __init__(self, args):
		args.setdefault("name", "PairedContextualIndex")
		super(PairedContextualIndex, self).__init__(args)

	def index(self, i, j, t, args):
		gap_, var_ = self.gap(i, args, j=j), self.variance(i, t, args, j=j)
		return gap_+var_

	def confidence_interval(self, i, t, args):
		mean_, var_ = self.gap(i, args), self.variance(i, t, args)
		return [mean_ - var_, mean_ + var_] 
