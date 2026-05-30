#!/usr/bin/env python3

import bees_algorithm as ba
import math, time, copy, multiprocessing, sys

__author__      = 'Luca Baronti'
__maintainer__  = 'Luca Baronti'
__license__     = 'GPLv3'
__version__     = '1.0.1'

def threadedGenerateScout(score_function,range_min,range_max,stlim,initial_ngh):
	bee=ba.Bee(range_min,range_max,stlim,initial_ngh,isForager=False,centre=None)
	bee.score=score_function(bee.values)
	return bee

def threadedGenerateForagers(score_function,site,n_foragers):
	foragers=[]
	for _ in range(n_foragers):
		bee=site.generateForager()
		bee.score=score_function(bee.values)
		foragers+=[bee]
	return foragers

def threadedLocalSearch(thread_index,current_site,score_function,n_foragers,range_min,range_max,stlim,initial_ngh,shrink_factor):
	if current_site.ttl==0: # abandon the site
		# generate n_foragers scouts and take the best one to replace the current site
		scouts=[threadedGenerateScout(score_function,range_min,range_max,stlim,initial_ngh) for _ in range(n_foragers)]
		scouts.sort(reverse=True)
		current_site=copy.deepcopy(scouts[0])
	else:
		foragers=threadedGenerateForagers(score_function,current_site,n_foragers)
		best_forager=ba.BeesAlgorithm._argmax(foragers)
		if best_forager.score > current_site.score:
			current_site=copy.deepcopy(best_forager)
			current_site.ttl=stlim	# site abandonment (reset ttl)
		else:
			current_site.ttl-=1	# site abandonment (reduce ttl)
			current_site.ngh=[x*(1.0 - shrink_factor) for x in current_site.ngh] # neighborhood shrinking
	return (thread_index, current_site)

def threadedFullLocalSearch(lock,n_iterations_global,best_solution,stop_criteria,score_function,n_foragers,range_min,range_max,stlim,initial_ngh,shrink_factor):
	# generate n_foragers scouts and take the best one to replace the current site
	scouts=[threadedGenerateScout(score_function,range_min,range_max,stlim,initial_ngh) for _ in range(n_foragers)]
	scouts.sort(reverse=True)
	current_site=copy.deepcopy(scouts[0])
	n_iterations=0
	n_abandons=0
	while True:
		if current_site.ttl==0: # abandon the site
			n_abandons+=1
			# check stop criteria
			lock.acquire()
			if best_solution['score']==None or current_site.score > best_solution['score']:
				best_solution['score']=current_site.score
				best_solution['position']=copy.deepcopy(current_site.values)
			n_iterations_global.value=max(n_iterations,n_iterations_global.value)
			if("max_iteration" in stop_criteria and n_iterations_global.value>=stop_criteria["max_iteration"]) or ("max_score" in stop_criteria and best_solution["score"] >= stop_criteria["max_score"]):
				lock.release()
				return 	# conclude the training
			lock.release()
			# generate n_foragers scouts and take the best one to replace the current site
			scouts=[threadedGenerateScout(score_function,range_min,range_max,stlim,initial_ngh) for _ in range(n_foragers)]
			scouts.sort(reverse=True)
			current_site=copy.deepcopy(scouts[0])
			n_iterations=0
		else:
			foragers=threadedGenerateForagers(score_function,current_site,n_foragers)
			best_forager=ba.BeesAlgorithm._argmax(foragers)
			if best_forager.score > current_site.score:
				current_site=copy.deepcopy(best_forager)
				current_site.ttl=stlim	# site abandonment (reset ttl)
				if("max_score" in stop_criteria and current_site.score >= stop_criteria["max_score"]):
					# the optimisation must be terminated (it's more difficult to do the same with the n_iterations)
					lock.acquire()
					if best_solution['score']==None or current_site.score > best_solution['score']:
						best_solution['score']=current_site.score
						best_solution['position']=copy.deepcopy(current_site.values)
					n_iterations_global.value=max(n_iterations,n_iterations_global.value)
					lock.release()
					return 	# conclude the training
			else:
				current_site.ttl-=1	# site abandonment (reduce ttl)
				current_site.ngh=[x*(1.0 - shrink_factor) for x in current_site.ngh] # neighborhood shrinking
		n_iterations+=1


class ParallelBeesAlgorithm(ba.BeesAlgorithm):
	def __init__(self,score_function,range_min,range_max,ns=10,nb=5,ne=1,nrb=10,nre=15,stlim=10,initial_ngh=None,shrink_factor=.2,useSimplifiedParameters=False,n_processes=multiprocessing.cpu_count()):
		ba.BeesAlgorithm.__init__(self,score_function,range_min,range_max,ns=ns,nb=nb,ne=ne,nrb=nrb,nre=nre,stlim=stlim,initial_ngh=initial_ngh,shrink_factor=shrink_factor,useSimplifiedParameters=useSimplifiedParameters)
		self.pool=multiprocessing.Pool(n_processes)
		self.n_foragers_list=[self.nre]*self.ne + [self.nrb]*(self.nb - self.ne)
		
	def __del__(self):
		self.pool.close()

	def _performLocalSearches(self):
		threads=[self.pool.apply_async(threadedLocalSearch, 
																	args=(i,self.current_sites[i],self.score_function,self.n_foragers_list[i],self.range_min,self.range_max,self.stlim,self.initial_ngh,
																				self.shrink_factor)) 
						for i in range(self.nb)]
		for t in threads:
			threads_result=t.get()
			self.current_sites[threads_result[0]]=threads_result[1]

class FullyParallelBeesAlgorithm(ba.BeesAlgorithm):
	def __init__(self,score_function,range_min,range_max,nb=5,nrb=10,stlim=10,initial_ngh=None,shrink_factor=.2,useSimplifiedParameters=False,n_processes=multiprocessing.cpu_count()):
		ba.BeesAlgorithm.__init__(self,score_function,range_min,range_max,ns=0,nb=nb,ne=0,nrb=nrb,nre=0,stlim=stlim,initial_ngh=initial_ngh,shrink_factor=shrink_factor,useSimplifiedParameters=useSimplifiedParameters)
		n_processes=min(n_processes,nb)
		self.pool=multiprocessing.Pool(n_processes)
		self.manager=multiprocessing.Manager()
		self.threads_lock=self.manager.Lock()
		self.termination_lock=self.manager.Lock()
		self.termination_lock.acquire()
		
	def __del__(self):
		self.pool.close()

	def performSingleStep(self):
		raise NotImplementedError("Can't run this parallel version of the Bees Algorithm one step at time.")

	def performFullOptimisation(self,max_iteration=None,max_score=None,verbose=0):	
		if max_iteration==None and max_score==None:
			raise ValueError("Called performFullOptimisation without a stop criteria")
		if max_iteration!=None and max_iteration<0:
			raise ValueError("The maximum number of iterations can't be negative")
		n_iterations=self.manager.Value('i',0)
		best_solution=self.manager.dict({'position':None,'score':None},lock=False)
		stop_criteria=self.manager.dict(lock=False)
		if max_iteration!=None:
			stop_criteria["max_iteration"]=max_iteration
		if max_score!=None:
			stop_criteria["max_score"]=max_score
		threads=[]
		for i in range(self.nb):
			threads+=[self.pool.apply_async(threadedFullLocalSearch, 
														args=(self.threads_lock,n_iterations,best_solution,stop_criteria,self.score_function,self.nrb,self.range_min,self.range_max,self.stlim,self.initial_ngh,self.shrink_factor), 
								)]
		# self.pool.close()
		self.pool.join()
		s=self._generateScout()
		s.values=best_solution["position"]
		s.score=best_solution["score"]
		self.best_solution=s
		return n_iterations.value, self.best_solution.score
