#!/usr/bin/env python3

import unittest as ut
import sys, logging
import benchmark_functions as bf
from computational_stopwatch import Stopwatch
import numpy as np
from bees_algorithm.bees_algorithm import BeesAlgorithm
from bees_algorithm.bees_algorithm_parallel_algorithm import ParallelBeesAlgorithm, FullyParallelBeesAlgorithm
from bees_algorithm.bees_algorithm_parallel_testing import BeesAlgorithmTester

N_RUNS 												= 100
MAX_ITERATIONS								= 5000
SCORE_THRESHOLD 							= 0.001
ITERATIONS_MEAN_THRESHOLD 		= 1.4
TEST_STD_DEV 									= False # removing all the std_dev tests since they can be wildely variable
ITERATIONS_STD_DEV_THRESHOLD 	= 1.7
ESPECTED_PARALLEL_GAIN				= 0.9

Griewank_bees_parameters=	{'ns':0,	'nb':18,	'ne':1,	'nrb':5,	'nre':10,	'stlim':5, 'useSimplifiedParameters':True}
Ackley_bees_parameters=		{'ns':0,	'nb':8,		'ne':1,	'nrb':10,	'nre':20,	'stlim':5, 'useSimplifiedParameters':True}
Easom_bees_parameters=		{'ns':0,	'nb':14,	'ne':1,	'nrb':5,	'nre':30,	'stlim':10, 'useSimplifiedParameters':True}
Schwefel_bees_parameters=	{'ns':0,	'nb':14,	'ne':1,	'nrb':5,	'nre':30,	'stlim':10, 'useSimplifiedParameters':True}

functions_list = [
	{'function': bf.Schwefel(n_dimensions=2, opposite=True), 'esp_iter_mean': 44.58 ,'esp_iter_std_dev': 13.64, 'parameters': Schwefel_bees_parameters},
	{'function': bf.Easom(opposite=True), 'esp_iter_mean': 38.28 ,'esp_iter_std_dev': 4.94, 'parameters': Easom_bees_parameters},
	{'function': bf.Ackley(n_dimensions=10, opposite=True), 'esp_iter_mean': 128.82 ,'esp_iter_std_dev': 29.77, 'parameters': Ackley_bees_parameters},
	{'function': bf.Griewank(n_dimensions=10, opposite=True), 'esp_iter_mean': 2659.06 ,'esp_iter_std_dev': 1889.61, 'parameters': Griewank_bees_parameters},
]

functions_running_time = {}

def build_algorithm(alg_type, function_to_test, ba_parameters):
	lower_bound, upper_bound = function_to_test.suggested_bounds()
	if alg_type in ['BeesAlgorithm', 'ParallelBeesAlgorithm']:
		return BeesAlgorithm(function_to_test,
										lower_bound, upper_bound,
										**ba_parameters
										)
	elif alg_type == 'FullyParallelBeesAlgorithm':
		return FullyParallelBeesAlgorithm(function_to_test,
										lower_bound, upper_bound,
										nb=ba_parameters['nb'], nrb=ba_parameters['nrb'], stlim=ba_parameters['stlim'], useSimplifiedParameters=True
										)
	else:
		raise ValueError(f"Unkown type of algorithm {alg_type}")

def test_algorithm(alg_type, function_to_test, ba_parameters):
	global functions_running_time
	results = []
	s = Stopwatch()
	for i in range(N_RUNS):
		alg = build_algorithm(alg_type, function_to_test, ba_parameters)
		it, _ = alg.performFullOptimisation(max_iteration=MAX_ITERATIONS, max_score=function_to_test.maximum().score - SCORE_THRESHOLD)
		results+=[it]
	if alg_type == 'BeesAlgorithm' and function_to_test.name() not in functions_running_time:
		functions_running_time[function_to_test.name()] = s.get_elapsed_time()
	return np.array(results)


class TestBeesAlgorithm(ut.TestCase):
	def test_functions(self):
		for function_info in functions_list:
			test_function = function_info['function']
			results = test_algorithm('BeesAlgorithm', test_function, function_info['parameters'])
			iterations_mean, iterations_mean = results.mean(), results.std()
			if results.mean()>function_info['esp_iter_mean']*ITERATIONS_MEAN_THRESHOLD:
				self.fail(f"Testing function {test_function.name()} mean iteration (out of {N_RUNS} runs) is {results.mean()} which is greater than {int(ITERATIONS_MEAN_THRESHOLD*100)}% the espected mean of {function_info['esp_iter_mean']}")
			if TEST_STD_DEV and results.std()>function_info['esp_iter_std_dev']*ITERATIONS_STD_DEV_THRESHOLD: 
				self.fail(f"Testing function {test_function.name()} std dev of the number of iterations (out of {N_RUNS} runs) is {results.std()} which is greater than {int(ITERATIONS_STD_DEV_THRESHOLD*100)}% the std deviation of {function_info['esp_iter_std_dev']}")
			
class TestParallelBeesAlgorithm(ut.TestCase):
	def test_functions_on_parallel(self):
		global functions_running_time
		for function_info in functions_list:
			test_function = function_info['function']
			if test_function not in ['Ackley', 'Griewank']: # if the function is too simple the parallelisation is derimental
				continue
			s = Stopwatch()
			results = test_algorithm('ParallelBeesAlgorithm', test_function, function_info['parameters'])
			tpba = s.get_elapsed_time()
			iterations_mean, iterations_mean = results.mean(), results.std()
			if results.mean()>function_info['esp_iter_mean']*ITERATIONS_MEAN_THRESHOLD:
				self.fail(f"Testing function {test_function.name()} mean iteration (out of {N_RUNS} runs) is {results.mean()} which is greater than {int(ITERATIONS_MEAN_THRESHOLD*100)}% the espected mean of {function_info['esp_iter_mean']}")
			if TEST_STD_DEV and results.std()>function_info['esp_iter_std_dev']*ITERATIONS_STD_DEV_THRESHOLD:
				self.fail(f"Testing function {test_function.name()} std dev of the number of iterations (out of {N_RUNS} runs) is {results.std()} which is greater than {int(ITERATIONS_STD_DEV_THRESHOLD*100)}% the std deviation of {function_info['esp_iter_std_dev']}")
			if test_function.name() not in functions_running_time:
				test_algorithm('BeesAlgorithm', test_function, function_info['parameters']) # functions_running_time will be updated inside this function
			tba = functions_running_time[test_function.name()]
			if tpba>tba*ESPECTED_PARALLEL_GAIN:
				self.fail(f"Testing function {test_function.name()} the parallel version of the BA took {Stopwatch.time_from_seconds(tpba)} which is more than {int(ESPECTED_PARALLEL_GAIN*100)}% of the vanilla BA time of {Stopwatch.time_from_seconds(tba)}")
	
	def test_functions_on_fully_parallel(self):
		global functions_running_time
		for function_info in functions_list:
			test_function = function_info['function']
			if test_function not in ['Ackley', 'Griewank']: # if the function is too simple the parallelisation is derimental
				continue
			s = Stopwatch()
			results = test_algorithm('FullyParallelBeesAlgorithm', test_function, function_info['parameters'])
			tpba = s.get_elapsed_time()
			iterations_mean, iterations_mean = results.mean(), results.std()
			if results.mean()>function_info['esp_iter_mean']*ITERATIONS_MEAN_THRESHOLD:
				self.fail(f"Testing function {test_function.name()} mean iteration (out of {N_RUNS} runs) is {results.mean()} which is greater than {int(ITERATIONS_MEAN_THRESHOLD*100)}% the espected mean of {function_info['esp_iter_mean']}")
			if TEST_STD_DEV and results.std()>function_info['esp_iter_std_dev']*ITERATIONS_STD_DEV_THRESHOLD:
				self.fail(f"Testing function {test_function.name()} std dev of the number of iterations (out of {N_RUNS} runs) is {results.std()} which is greater than {int(ITERATIONS_STD_DEV_THRESHOLD*100)}% the std deviation of {function_info['esp_iter_std_dev']}")
			if test_function.name() not in functions_running_time:
				test_algorithm('BeesAlgorithm', test_function, function_info['parameters']) # functions_running_time will be updated inside this function
			tba = functions_running_time[test_function.name()]
			if tpba>tba*ESPECTED_PARALLEL_GAIN:
				self.fail(f"Testing function {test_function.name()} the fully parallel version of the BA took {Stopwatch.time_from_seconds(tpba)} which is more than {int(ESPECTED_PARALLEL_GAIN*100)}% of the vanilla BA time of {Stopwatch.time_from_seconds(tba)}")

	def test_functions_on_parallel_testing(self):
		global functions_running_time
		for function_info in functions_list:
			test_function = function_info['function']
			if test_function not in ['Ackley', 'Griewank']: # if the function is too simple the parallelisation is derimental
				continue
			lower_bound, upper_bound = test_function.suggested_bounds()
			s = Stopwatch()
			tester=BeesAlgorithmTester(	test_function,
															lower_bound, upper_bound,
															**function_info['parameters'])
			tester.run_tests(n_tests=N_RUNS, max_iteration=MAX_ITERATIONS, max_score=test_function.maximum().score - SCORE_THRESHOLD,verbose=False)
			tpba = s.get_elapsed_time()
			results = np.array([x[0] for x in tester.results])
			iterations_mean, iterations_mean = results.mean(), results.std()
			if results.mean()>function_info['esp_iter_mean']*ITERATIONS_MEAN_THRESHOLD:
				self.fail(f"Testing function {test_function.name()} mean iteration (out of {N_RUNS} runs) is {results.mean()} which is greater than {int(ITERATIONS_MEAN_THRESHOLD*100)}% the espected mean of {function_info['esp_iter_mean']}")
			if TEST_STD_DEV and results.std()>function_info['esp_iter_std_dev']*ITERATIONS_STD_DEV_THRESHOLD:
				self.fail(f"Testing function {test_function.name()} std dev of the number of iterations (out of {N_RUNS} runs) is {results.std()} which is greater than {int(ITERATIONS_STD_DEV_THRESHOLD*100)}% the std deviation of {function_info['esp_iter_std_dev']}")
			if test_function.name() not in functions_running_time:
				test_algorithm('BeesAlgorithm', test_function, function_info['parameters']) # functions_running_time will be updated inside this function
			tba = functions_running_time[test_function.name()]
			if tpba>tba*ESPECTED_PARALLEL_GAIN:
				self.fail(f"Testing function {test_function.name()} the testing parallel version of the BA took {Stopwatch.time_from_seconds(tpba)} which is more than {int(ESPECTED_PARALLEL_GAIN*100)}% of the vanilla BA time of {Stopwatch.time_from_seconds(tba)}")


if __name__=='__main__':
	logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
	ut.main()

