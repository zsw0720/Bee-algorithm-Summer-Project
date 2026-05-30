#!/usr/bin/env python3

import copy, logging
import numpy as np

__author__      = 'Luca Baronti'
__maintainer__  = 'Luca Baronti'
__license__     = 'GPLv3'
__version__     = '1.0.2'

class Bee(object):
	def __init__(self, range_min, range_max, ttl, ngh, isForager=False, centre=None): # isForager==False means that it's a (global) scout bee
		self.ttl = ttl
		self.ngh = ngh
		self.values = []
		self.range_min = range_min
		self.range_max = range_max
		if centre is None:
			centre=[(range_max[i] + range_min[i])/2.0 for i in range(len(range_min))] # middle point
		self.score = None
		if isForager:
			self.initialiseValues(ngh, centre=centre)
		else:
			self.initialiseValues([1.0]*len(range_min), centre=centre)

	def initialiseValues(self, ngh, centre):
		self.values = np.zeros(len(self.range_min))
		for i in range(len(self.range_min)):
			v = (self.range_max[i] - self.range_min[i])*.5
			self.values[i] = np.random.uniform(-v,v)*ngh[i] + centre[i]
			self.values[i] = min(self.values[i], self.range_max[i])
			self.values[i] = max(self.values[i], self.range_min[i])
	
	def generateForager(self):
		return Bee(self.range_min, self.range_max, self.ttl, self.ngh, isForager=True, centre=self.values)

	def __str__(self):
		return "S="+str(self.score)+" "+str(self.values)

	def __lt__(self, other):
		return self.score < other.score

class BeesAlgorithm(object):
	def __init__(self, score_function, range_min, range_max, ns=10, nb=5, ne=1, nrb=10, nre=15, stlim=10, initial_ngh=None, shrink_factor=.2, useSimplifiedParameters=False):
		if useSimplifiedParameters:
			self.ns = ns
		else:
			self.ns = ns - nb
		self.nb = nb
		self.ne = ne
		self.nrb = nrb
		self.nre = nre
		self.stlim = stlim			
		if initial_ngh is None:
			self.initial_ngh = np.ones(len(range_min))
		else:
			self.initial_ngh = np.array(initial_ngh)
		self.shrink_factor = shrink_factor
		self.range_max = range_max
		self.range_min = range_min
		self.score_function = score_function
		self.keep_bees_trace = False # this is used only for visualisation purposes
		self._validate()
		# initialise the first bees
		self.current_sites = []
		self.best_solution = None
		self._validate()
		self._initialise_solutions()

	# performs sanity checks and raise an exception if some initialisation parameters are wrong
	def _validate(self):
		if len(self.range_min)!=len(self.range_max):
			raise ValueError("The sizes of the lower and upper bounds don't match ("+str(len(self.range_min))+"!="+str(len(self.range_max))+")")
		if len(self.initial_ngh)!=len(self.range_max):
			raise ValueError("The size of the initial neighborhood doesn't match with the size of the lower and upper bounds ("+str(len(self.initial_ngh))+"!="+str(len(self.range_max))+")")
		for i in range(len(self.range_min)):
			if self.range_min[i]>=self.range_max[i]:
				raise ValueError("The "+str(i+1)+"-th value of the lower bound is greater or equals the respective value of the upper ("+str(self.range_min[i])+">="+str(self.range_max[i])+")")
			if self.initial_ngh[i]<0.0 or self.initial_ngh[i]>1.0:
				raise ValueError("The "+str(i+1)+"-th value of the initial neighborhood is not in the [0,1] range ("+str(self.initial_ngh[i])+")")
		if self.ne>self.nb:
			raise ValueError("The number of elite sites is higher of the number of best sites ("+str(self.ne)+">"+str(self.nb)+")")
		if self.shrink_factor<0.0 or self.shrink_factor>1.0:
			raise ValueError("The shrink factor is not in the [0,1] range ("+str(self.shrink_factor)+")")
			
	# Returns the number of checks in the hypothesis space in initialisation and for each single iteration
	def getChecksNumberPerIteration(self):
		return self.ns + (self.nb - self.ne)*self.nrb + self.ne*self.nre
	
	# Performs a single step of the algorithm, modifying best_solution and current_sites accordingly
	def performSingleStep(self):
		if self.keep_bees_trace: # this is used only for visualisation purposes
			self.to_save_best_sites=[copy.deepcopy(x) for x in self.current_sites]
			self.to_save_foragers=[]
		self._performLocalSearches()
		# Add the scouts
		self.current_sites+=[self._generateScout() for _ in range(self.ns)]
		# Sort and take only the best ones
		self.current_sites.sort(reverse=True)
		self.current_sites=self.current_sites[:self.nb]
		if self.current_sites[0].score > self.best_solution.score:
			self.best_solution=copy.deepcopy(self.current_sites[0])
		return self.best_solution.score

	# Performs a full optimisation, terminating when either one of the stop criteria is met
	# it returns the number of iterations performed and the best score found
	def performFullOptimisation(self, max_iteration=None, max_score=None, verbose=0):
		if max_iteration is None and max_score is None:
			raise ValueError("Called performFullOptimisation without a stop criteria")
		if max_iteration is not None and max_iteration < 0:
			raise ValueError("The maximum number of iterations can't be negative")
		iteration=0
		while (max_iteration is None or iteration<max_iteration) and (max_score is None or self.best_solution.score < max_score):
			self.performSingleStep()
			iteration+=1
			if verbose>0:
				print("Iteration:",iteration,"Best:",self.best_solution,end='')
				if verbose == 1:
					print('')
				else:
					print(" All:",[str(x) for x in self.current_sites])
		return iteration, self.best_solution.score

	def _performLocalSearches(self):
		for i in range(len(self.current_sites)):
			if i<self.ne: # it's a elite site
				n_foragers=self.nre
			else: # it's a best site
				n_foragers=self.nrb
			self._localSearch(i,n_foragers)

	def _localSearch(self, index, n_foragers):
		if self.current_sites[index].ttl==0: # abandon the site
			# generate n_foragers scouts and take the best one to replace the current site
			scouts=[self._generateScout() for _ in range(n_foragers)]
			scouts.sort(reverse=True)
			self.current_sites[index]=copy.deepcopy(scouts[0])
		else:
			foragers=self._generateForagers(self.current_sites[index],n_foragers)
			if self.keep_bees_trace: # this is used only for visualisation purposes
				self.to_save_foragers+=[foragers]
			best_forager=BeesAlgorithm._argmax(foragers)
			if best_forager.score > self.current_sites[index].score:
				self.current_sites[index]=copy.deepcopy(best_forager)
				self.current_sites[index].ttl=self.stlim	# site abandonment (reset ttl)
			else:
				self.current_sites[index].ttl-=1	# site abandonment (reduce ttl)
				self.current_sites[index].ngh=[x*(1.0 - self.shrink_factor) for x in self.current_sites[index].ngh] # neighborhood shrinking

	def _initialise_solutions(self):
		self.current_sites=[self._generateScout() for _ in range(self.getChecksNumberPerIteration())]
		self.current_sites.sort(reverse=True)
		self.current_sites=self.current_sites[:self.nb]
		self.best_solution=self.current_sites[0]
	
	def _generateScout(self):
		bee = Bee(self.range_min, self.range_max, self.stlim, self.initial_ngh, isForager=False, centre=None)
		bee.score = self.score_function(bee.values)
		return bee

	def _generateForager(self, site):
		bee = site.generateForager()
		bee.score = self.score_function(bee.values)
		return bee

	def _generateForagers(self,site,n_foragers):
		return [self._generateForager(site) for _ in range(n_foragers)]

	@staticmethod
	def _argmax(solutions):
		solution_best = None
		for sol in solutions:
			if solution_best is None or sol.score > solution_best.score:
				solution_best = sol
		return solution_best


	def visualize_iteration_steps(self):
			try:
					import matplotlib.pyplot as plt
			except ImportError:
					logging.error("In order to visualise the steps of the LORRE Algorithm, the following libraries are required: numpy, mpl_toolkits and matplotlib")
					return

			self.keep_bees_trace = True
			if len(self.range_min) != 2 or len(self.range_max)!=2:
					logging.error("In order to visualise the steps of the LORRE Algorithm the score funciton must be defined in 2 dimensions")
			print(self.range_min)
			x = np.linspace(self.range_min[0], self.range_max[0], 50)
			y = np.linspace(self.range_min[1], self.range_max[1], 50)

			X, Y = np.meshgrid(x, y)
			Z = np.asarray([[self.score_function([X[i][j], Y[i][j]])
			                    for j in range(len(X[i]))] for i in range(len(X))])
			p_size = (self.range_max[0] - self.range_min[0])*.01
			fig = plt.figure()
			iteration = 0
			ax = plt.axes(projection='3d')
			ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none',alpha=.3)
			ax.set_xlabel('x')
			ax.set_ylabel('y')
			ax.set_zlabel('z')
			ax.view_init(60, 35)
			while True:
					iteration += 1
					self.performSingleStep()
					fig.canvas.manager.set_window_title("Bees Algorithm Iteration "+str(iteration))
					fig.suptitle("Best Solution "+str(self.best_solution))
					points_x, points_y = [], []
					colors, sizes = [], []
					for b in self.to_save_best_sites:
							points_x += [b.values[0]]
							points_y += [b.values[1]]
							colors   += ['blue']
							sizes    += [p_size*2.0]
					for fs in self.to_save_foragers:
							for f in fs:
									points_x += [f.values[0]]
									points_y += [f.values[1]]
									colors   += ['purple']
									sizes    += [p_size]
					points_z = [self.score_function([points_x[i], points_y[i]])
					               for i in range(len(points_x))]
					points = ax.scatter(points_x, points_y, points_z, c=colors, s=sizes)
					fig.canvas.draw()
					fig.canvas.flush_events()
					fig.show()
					input("Press any key to continue..")
					points.remove()
