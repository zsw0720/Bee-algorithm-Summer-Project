#!/usr/bin/env python3

import bees_algorithm
import time, multiprocessing

__author__      = 'Luca Baronti'
__maintainer__  = 'Luca Baronti'
__license__     = 'GPLv3'
__version__     = '1.0.1'

def threaded_run(ba_instance,max_iteration,max_score,verbose):
	iteration, _ =ba_instance.performFullOptimisation(max_iteration=max_iteration,max_score=max_score,verbose=0)
	result=ba_instance.best_solution
	if verbose:
		print("Run ended, max iteration",iteration,"solution",result)
	return (iteration, result)

class BeesAlgorithmTester(object):
	def __init__(self,score_function,range_min,range_max,ns=10,nb=5,ne=1,nrb=10,nre=15,stlim=10,initial_ngh=None,shrink_factor=.2,useSimplifiedParameters=False):
		self.score_function=score_function
		self.range_min=range_min
		self.range_max=range_max
		self.ns=ns
		self.nb=nb
		self.ne=ne
		self.nrb=nrb
		self.nre=nre
		self.stlim=stlim
		self.initial_ngh=initial_ngh
		self.shrink_factor=shrink_factor
		self.useSimplifiedParameters=useSimplifiedParameters
		self.results=[] # will contains all the results in the form of (iteration_reached,best_solution)
		self.iterations5values=None
		self.scores5values=None

	def run_tests(self,n_tests,n_processes=multiprocessing.cpu_count(),max_iteration=None,max_score=None,verbose=True):
		if max_iteration==None and max_score==None:
			raise ValueError("No stop criteria set")
		algs=[]
		pool=multiprocessing.Pool(n_processes)
		for i in range(n_tests):
			a=bees_algorithm.BeesAlgorithm(	self.score_function,
																			self.range_min,self.range_max,
																			ns=self.ns,nb=self.nb,ne=self.ne,nrb=self.nrb,
																			nre=self.nre,stlim=self.stlim,useSimplifiedParameters=self.useSimplifiedParameters)		
			algs+=[a]
		threads=[pool.apply_async(threaded_run, args=(algs[i],max_iteration,max_score,verbose)) for i in range(n_tests)]
		self.results=[t.get() for t in threads]
		pool.close()
		iterations=[r[0] for r in self.results]
		scores=[r[1].score for r in self.results]
		iterations.sort()
		scores.sort()
		self.iterations5values=(iterations[0],iterations[int(n_tests*.25)],iterations[int(n_tests*.5)],iterations[int(n_tests*.75)],iterations[-1])
		self.scores5values=(scores[0],scores[int(n_tests*.25)],scores[int(n_tests*.5)],scores[int(n_tests*.75)],scores[-1])



