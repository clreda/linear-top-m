#coding: utf-8

import numpy as np

import utils

####################################################
## Learners                                       ##
####################################################

# Learner structure
class Learner(object):
	def __init__(self, K):
		self.K = K
		## Arm playing mixed strategy
		self.p = np.zeros(self.K)
	def act(self):
		raise NotImplemented
	def incur(self, w):
		raise NotImplemented

# Follow the Leader learner
class FTL(Learner):
	def __init__(self, K):
		super(FTL, self).__init__(K)
	def act(self):
		p = np.asarray(self.p == np.min(self.p), dtype=float)
		p /= np.sum(p)
		return p
	def incur(self, w):
		self.p += w

# AdaHedge learner (the one used in LinGame)
class AdaHedge(Learner):
	def __init__(self, K):
		self.delta = 0.01
		super(AdaHedge, self).__init__(K)
	def act(self):
		eta = np.log(self.K)/float(self.delta)
		p = np.exp(-eta*(self.p-np.min(self.p)))
		p /= np.sum(p)
		return p
	def incur(self, w):
		p = self.act()
		eta = np.log(self.K)/float(self.delta)
		self.p += w
		m = np.min(w)-1./eta*np.log(p.T.dot(np.exp(-eta*(w-np.min(w)))))
		assert m != float("inf") and p.T.dot(w) >= m-1e-7
		self.delta += p.T.dot(w)-m

# Lazy Mirror Descent/Fixed share learner
class FixedShare(Learner):
	def __init__(self, K, S=1):
		self.K = K
		self.p = 1./float(self.K)*np.ones(self.K)
		self.t = 0
		self.S = S
	def act(self):
		return self.p
	def incur(self, w):
		self.t += 1
		eta = np.sqrt(np.log(self.K)/float(self.t))/float(self.S)
		self.p *= np.exp(-eta*(w-np.min(w)))
		self.p /= np.sum(self.p)
		gamma = 1./float(4*np.sqrt(self.t))
		self.p = (1-gamma)*self.p+gamma/float(self.K)
