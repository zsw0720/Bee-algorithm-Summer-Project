# BeesAlgorithm - A Python Implementation

This repository contains a Bees Algorithm implementation in Python 3.

The aim is to make available to everyone an implementation, built with the minimal number of dependencies, which can be easily integrated in larger projects as well as used out-of-the-box to solve specific problems.

The Bees Algorithm is an intelligent optimization technique belonging to the swarm algorithms field.
Given a parametric objective function, the goal of the algorithm is to find the parameter values that maximise/minimise the output of the objective function.

Many real-world problems can be modeled as the optimisation of a parametric objective function, therefore effective algorithms to handle this kind of problems are of primary importance in many fields.
The Bees Algorithm performs simultaneus aggressive local searches around the most promising parameter settings of the objective function.
The algorithm is proven to outperform other intelligent optimisation techniques in many benchmark functions<sup>[3][4][5]</sup> as well as real world problems.

On top of this Python version, implmentations of the bees Algorithm in [C++](https://gitlab.com/bees-algorithm/bees_algorithm_cpp) and [Matlab](https://gitlab.com/bees-algorithm/bees_algorithm_matlab) are also available in the respective repositories. 

The main steps of the Bees Algorithm will be described in the next section. For more information please refer to the [official Bees Algorithm website](http://beesalgorithmsite.altervista.org) and the [wikipedia page](https://en.wikipedia.org/wiki/Bees_algorithm). If you are interested in a detailed analysis of the algorithm, and the properties of its search strategy,  please refer to this paper<sup>[1]</sup>:

- Luca Baronti, Marco Castellani, and Duc Truong Pham. "[An Analysis of the Search Mechanisms of the Bees Algorithm.](
https://doi.org/10.1016/j.swevo.2020.100746)" Swarm and Evolutionary Computation 59 (2020): 100746.

If you are using this implementation of the Bees Algorithm for your research, feel free to cite this work in your paper using the following BibTex entry:
```bibtex
@article{baronti2020analysis,
  title={An Analysis of the Search Mechanisms of the Bees Algorithm},
  author={Baronti, Luca and Castellani, Marco and Pham, Duc Truong},
  journal={Swarm and Evolutionary Computation},
  volume={59},
  pages={100746},
  year={2020},
  publisher={Elsevier},
  doi={10.1016/j.swevo.2020.100746},
  url={https://doi.org/10.1016/j.swevo.2020.100746}
}
```

# Installation
This module is available on [pip](https://pypi.org/project/bees-algorithm) and can be installed as follows:
```sh
$ pip3 install bees_algorithm
```


# Introduction on the Bees Algorithm
<img src="https://images.pexels.com/photos/1043059/pexels-photo-1043059.jpeg"
     align="left"
     width="333" height="250" />

The Bees Algorithm is a nature-inspired search method that mimics the foraging behaviour of honey bees. It was created by [Prof. D.T. Pham](https://www.birmingham.ac.uk/staff/profiles/mechanical/pham-duc.aspx) and his co-workers in 2005<sup>[2]</sup>, and described in its standard formulation by Pham and Castellani<sup>[3]</sup>.

The algorithm uses a population of agents (artificial bees) to sample the solution space. A fraction of the population (scout bees) searches randomly for regions of high fitness (global search). The most successful scouts recruit a variable number of idle agents (forager bees) to search in the proximity of the fittest solutions (local search). Cycles of global and local search are repeated until an acceptable solution is discovered, or a given number of iterations have elapsed.

The standard version of the Bees Algorithm includes two heuristics: *neighbourhood shrinking* and *site abandonment*.
Using neighbourhood shrinking the size of the local search is progressively reduced when local search stops progressing on a given site.
The site abandonment procedure interrupts the search at one site after a given number of consecutive stagnation cycles, and restarts the local search at a randomly picked site.


<img src="pics/algorithm.png"
     align="right"
     width="300" height="300" />

The algorithm requires a number of parameters to be set, namely: the number of scout bees (*ns*), number of sites selected out of *ns* visited sites (*nb*), number of elite sites out of *nb* selected sites (*ne*), number of bees recruited for the best *ne* sites (*nre*), number of bees recruited for the other (*nb-ne*) selected sites (*nrb*).
The heuristics also require the set of the initial size of the patches (*ngh*) and the number of cycles after which a site is abandoned (*stlim*).
Finally, the stopping criterion must be defined. 

The algorithm starts with the *ns* scout bees being placed randomly in the search space and the main algorithm steps can be summarised as follows:


1. Evaluate the fitness of the population according the objective function;
2. Select the best *nb* sites for neighbourhood (local) search;
3. Recruit *nrb* forager bees for the selected sites (*nre* bees for the best *ne* sites) and evaluate their fitnesses;
4. Select the fittest bee from each local site as the new site centre;
5. If a site fails to improve in a single local search, its neighbourhood size is reduced (neighbourhood shrinking);
6. If a site fails to improve for *stlim* cycles, the site is abandoned (site abandonment);
7. Assign the remaining bees to search uniformly the whole search space and evaluate their fitnesses;
8. If the stopping criterion is not met, return to step 2;

# Usage
This repository offers two kinds of libraries. 
The library contained in *bees_algorithm.py* is a simple python implementation of the iterative Bees Algorithm.
The libraries contained in *bees_algorithm_parallel_algorithm.py* and *bees_algorithm_parallel_testing.py* offer parallel versions of the algorithm, which is implemented on two different levels of parallelisms.

### Guidelines
The implementations present here try to cover the following use cases:

- General use of the Bees Algorithm on a single test when only a limited number of iterations are expected: use *BeesAlgorithm* from *bees_algorithm.py*;
- Single test using a computationally expensive objective function, a large number of dimensions or iterations: use *ParallelBeesAlgorithm* from *bees_algorithm_parallel_algorithm.py*;
- Single test using an even more computationally expensive objective function, a larger number of dimensions or iterations: use *FullyParallelBeesAlgorithm* from *bees_algorithm_parallel_algorithm.py*;
- Multiple tests on the same objective function: use *BeesAlgorithmTester* from *bees_algorithm_parallel_testing.py*;

Please refer to the tests present in the following [test section](#tests) for the resons behind these guidelines.

## Iterative Version
To use the library just import it
```python
from bees_algorithm import BeesAlgorithm
```
then define your objective function (an hypesphere in this case)
```python
def hypersphere(x):
    return -sum([pow(xi,2) for xi in x])
```
and the search boundaries (in this case we are assuming a 4-dimensional search)
```python
search_boundaries=([-5,-5,-5,-5], [5,5,5,5])
```
This implementation of the bees algorithm will always try to find the solution that **maximize** the objective function.
Therefore, if you have to find the minimum of your function $`g(x)`$ simply implement the objective function $`f(x)=-g(x)`$.

The next step is create an instance of the algoritm:
```python
alg = BeesAlgorithm(hypersphere,search_boundaries[0],search_boundaries[1])
```
This will create an instance of the bees algorithm with default parameters (ns=10,nb=5,ne=1,nrb=10,nre=15,stlim=10,shrink_factor=.2).
To use a different set of parameters, it's sufficient to pass the right values in the constructor like that:
```python
alg = BeesAlgorithm(hypersphere,search_boundaries[0],search_boundaries[1],ns=0,nb=14,ne=1,nrb=5,nre=30,stlim=10)
```
This implementation of the bees algorithm use a simplified parameters formulation for the scout bees.
Normally the parameter *ns* indicates the number of best sites **and** the number of scout bees used for the global search at each iteration.
In this simpler formulation the parameter *ns* is the number of scout bees that are exclusively used for the global search, instead.
Therefore setting *ns=0* means that no scout bees are used for the global search.

In any case, it's possible to use the traditional parameters notation setting *useSimplifiedParameters=True* in the constructor.

To perform the optimisation is possible to do it iteratively calling:
```python
alg.performSingleStep()
```
which perform a single iteration of the algoritm.
Alternatively it's possible to perform the search all at once with:
```python
alg.performFullOptimisation(max_iteration=5000)
```
In this case, two different stop criteria can be used: *max_iteration* wil interrupt the optimisation when a certain number of iterations are reached and *max_score* will interrupt it when a certain score is reached.
If this method is used, at least one stop criterion must be specified.
Finally, it is possible to use both the stop criteria, in which case the algorithm will stop when the first one is reached.

To assess the state of the algorithm, as well as the final result, the following variable can be accessed:
```python
best = alg.best_solution
best.score      # is the score of the best solution (e.g. 0.0 in our case)
best.values     # are the coordinates of the best solution (e.g. [0.0, 0.0, 0.0, 0.0] in our case)
```
It is also possible to asses the score and values of all the current *nb* best sites accessing to the list *alg.current_sites*.

The variable *best_solution* contains the best solution found so far by the algorithm, which may or may not be included in the current best sites list *current_sites*.

## Parallel Versions
The parallel version exposes the data structures and functions to run the Bees Algorithm in parallel on two different levels:

- At the testing level, where a certain number of instances of the iterative Bees Algorithm are run in parallel on the same objective function;
- At the algorithm level, where a single instance of a parallel version of the Bees Algorithm is run on a certain objective function;

Despite using a prallel version of the algorithm in a parallel may looks like a good idea, the eccessive prolification of processes that will result would most likely impact negatively on the performance.
For this reason other kind of *mixed* parallelisms are not implemented here.
### Parallel Version: Testing Level
Being a sthocastic algorithm, the final result of the Bees Algorithm is not deterministic. 
Therefore, running multiple instances of the algorithm on the same problem is useful to assess the general quality of the Bees Algorithm, or of a certain selection of its parameters, on a given problem.

In this case (testing level) it is possible to import:
```python
from bees_algorithm import BeesAlgorithmTester
```
create an instance of the class *BeesAlgorithmTester* class similarly to how is done with the iterative version:
```python
tester = BeesAlgorithmTester(hypersphere,search_boundaries[0],search_boundaries[2])
```
and finally call the *run_tests* function:
```python
tester.run_tests(n_tests=50, max_iteration=5000)
```
which accepts the parameter *n_tests*, which defines how many test ought to be performed, the optional parameter *verbose*, useful to assess the completion of the single processes and the stop criteria used (similar to the *performFullOptimisation* function).

Optionallly, it is also possible to pass the argument *n_processes* to define the degree of parallelism (namely, how many processes are run at the same time).
If it is not present, the number of cores of the CPU will be used as degree of parallelism.
It is advisable to not use an high value for this argument, otherwise the performances may be *negatively* impacted.

Once the *run_tests* function is terminated the final results can be accessed using the *results* list of the instance, this way:
```python
res = tester.results
```
each element of the *results* list is a tuple *(iteration, best_solution)* representing the iteration reached and the best solution found for a run.

### Parallel Version: Algorithm Level
Some characteristics of the Bees Algorithm pose serious limitation on the design of an effective parallel version.
One of the most effective way to perform the algorithm in parallel is to run the sequence of sites local searches simultaneously.
However, mapping a site search search on a thread/process that perform the whole site search (up to the point the site is abandoned) can't be entirerly done due to some info that need to be shared between the sites.
For instance, at each iteration the algorithm must know:

- which *nb* sites are the best ones, to promote them to elite sites and give them more foragers;
- which *nb* sites need to be replaced by better solutions found in the global search (*ns*);
- how many local searches have been done so far, and what's the best solution, to assess the stop criteria;

To assess all these information, the parallel searches must stop-and-report to a central controller, reducing the parallelism performances greately.
Removing heuristics like the elite sites, the global scouts, and accepting some approximation on the stop criteria (i.e. updating the number of iterations and best score only when a site is abandoned) will remove this problem.
However these modifications will lead to a quite different algorithm than the standard one, so the two different approaches are implemented in this library.

#### Parallel Version: Algorithm Level - Partial
In this version, the Bees Algorithm works in parallel only in performing a single local search for all the sites.

That is, at each iteration, the *nb* sites perform a single local search in parallel.
In this way it's possible to control the other aspects of the algorithm keeping its original behaviour and, at the same time, have an improvment in the completion time to a certain extent.

To use this parallel version of the Bees Algorithm it is possible to import:
```python
from bees_algorithm import ParallelBeesAlgorithm
```
and create an instance of the class *ParallelBeesAlgorithm* class exactly how is done with the iterative version:
```python
alg = ParallelBeesAlgorithm(hypersphere,search_boundaries[0],search_boundaries[2])
```
where the constructors takes all the parameters of the iterative version, with the addition of the *n_processes* parameters, set as the number of cores of the current machine as default.
The algorithm instance can then be used exactly the same way as the iterative case.

#### Parallel Version: Algorithm Level - Full
In this version, *nb* sites perform separate searches in a higher degree of parallelism.

Under ideal circumnstances, *nb* threads are run in parallel, each one performing a search for a site.
When the site is abandoned, the associated thread will generate a new site and start a new search.
Threads communicate to eachothers only in assessing the stop criteria, mostly when a site is abandoned.
This will avoid overhead induced by accessing to shared resources with high frequency.

This version can potentially offers an higher performance Bees Algorithm, however it comes with the following limitations:

- No elite sites can be used;
- No global scouts can be used;
- The value of the final number of iterations may not be accurate (can be overestimed);

To use this parallel version of the Bees Algorithm it is possible to import the same file:
```python
from bees_algorithm import FullyParallelBeesAlgorithm
```
and create an instance of the class *FullyParallelBeesAlgorithm* class similarry as it is done with the iterative version:
```python
alg = FullyParallelBeesAlgorithm(hypersphere,search_boundaries[0],search_boundaries[2])
```
The key difference is that the constructor accept the following parameters: *score_function*, *range_min*, *range_max*, *nb*, *nrb*, *stlim*, *initial_ngh*, *shrink_factor*,*useSimplifiedParameters* and *n_processes*.

The number of threads used in this case will be:
```math
\min(nb, n\_processes)
```
The algorithm instance can then be used exactly the same way as the iterative case.

# Step-by-step Visualisation
A function that plot a visual rapresentation of the Bees Algorithm steps is included in this package.
The function *visualize_iteration_steps* can be called in lieu of *performFullOptimisation*. It accepts no parameters.

Here is an example using the *Schwefel* target function (see section below for details):
```python
import benchmark_functions as bf

b_func = bf.Schwefel(opposite=True)
suggested_lowerbound, suggested_upperbound = b_func.suggested_bounds()
schwefel_bees_parameters = {'ns':0, 'nb':14, 'ne':1, 'nrb':5, 'nre':30, 'stlim':10}

alg = BeesAlgorithm(b_func,
                    suggested_lowerbound, suggested_upperbound,
                    **schwefel_bees_parameters)

alg.visualize_iteration_steps()
```

# Tests
Tests on some common benchmark functions can be performed using the routines included in the library.

In order to do so, the [Python Benchmark Functions Library](https://gitlab.com/luca.baronti/python_benchmark_functions) must be used.
This can be installed as follows:
```sh
$ pip3 install benchmark_functions
```
This extra library is only needed to perform these tests, it's not necessary for other parts of the Bess Algorithm library.

In the installation directory is present a *test.py* script which contains a series of testing functions. These tests have the purpose of assessing the Bees Algorithm performances as well as a general sanity check.

## Test function *test_bees_algorithm*
This function will perform 50 runs of the iterative version of the Bees Algorithm on the [Schwefel](https://www.sfu.ca/~ssurjano/schwef.html), [Easom](https://www.sfu.ca/~ssurjano/easom.html), [Ackley](https://www.sfu.ca/~ssurjano/ackley.html), and the [Griewank](https://www.sfu.ca/~ssurjano/griewank.html)  functions, producing the following report:
```
Function Schwefel (expected results: mean~44.58 std_dev~13.64)
Run	Iteration	Score
==============================
0	39	837.965275496
5	37	837.965688486
10	32	837.964956329
15	30	837.965139015
20	45	837.965186794
25	37	837.965180098
30	67	837.964860797
35	41	837.965147787
40	31	837.965015294
45	34	837.964948223

Iterations Average 39.5 Standard Deviation 7.84155596805
Score Average 837.965237882 Standard Deviation 0.000287937127644
Elapsed time: 00:00:00

Function Easom (expected results: mean~38.28 std_dev~4.94)
Run	Iteration	Score
==============================
0	34	0.999930176221
5	39	0.999983161479
10	29	0.999411201819
15	28	0.999823474998
20	36	0.999195124497
25	38	0.999782416819
30	28	0.99956136641
35	36	0.999576114135
40	28	0.999352267829
45	34	0.999096989796

Iterations Average 33.26 Standard Deviation 4.18
Score Average 0.999622438808 Standard Deviation 0.000309312566353
Elapsed time: 00:00:00

Function Ackley (expected results: mean~128.82 std_dev~29.77)
Run	Iteration	Score
==============================
0	122	-0.000767782894592
5	111	-0.000942509624391
10	453	-0.000962877569986
15	119	-0.000793938600282
20	105	-0.00080786067877
25	122	-0.000958432201266
30	159	-0.000593911821916
35	117	-0.000946378270076
40	115	-0.000887289267987
45	112	-0.00088327532772

Iterations Average 138.12 Standard Deviation 68.7385306797
Score Average -0.000876801767429 Standard Deviation 0.000107520363268
Elapsed time: 00:00:08

Function Griewank (expected results: mean~2659.06 std_dev~1889.61)
Run	Iteration	Score
==============================
0	2360	-0.000590485619133
5	2072	-0.000924699852928
10	68	-0.000542138820798
15	1111	-0.000839167482014
20	2061	-0.000869748495223
25	4675	-0.000897005831677
30	4457	-0.000983333137394
35	1906	-0.000988599879426
40	918	-0.000853108837525
45	5000	-0.00739604033412

Iterations Average 2400.06 Standard Deviation 1702.20240171
Score Average -0.00174099693043 Standard Deviation 0.00228458642363
Elapsed time: 00:03:08

Elapsed time (overall): 00:03:18
```
The results are comparable to the ones achieved in the original paper<sup>[3]</sup>.

## Test function *test_parallel_algorithm*
The tests on the different parallel versions are performed only on the most complex functions considered, namely the [Ackley](https://www.sfu.ca/~ssurjano/ackley.html) and the [Griewank](https://www.sfu.ca/~ssurjano/griewank.html) functions.

The parallelism at application level can be tested with the following command:
```sh
$ python bees_algorithm_parallel_algorithm.py
```
which will produce the following result:
```
>>> The Partial Parallel Bees Algorithm will now be tested <<<
Function Ackley (expected results: mean~128.82 std_dev~29.77)
Run	Iteration	Score
==============================
0	112	-0.000966727923001
5	120	-0.000797724521061
10	107	-0.000866169613229
15	120	-0.000895930046074
20	114	-0.000944332810278
25	125	-0.000554040385375
30	131	-0.000910703005613
35	124	-0.000926084701039
40	115	-0.000992554270788
45	106	-0.000848826170178

Iterations Average 131.74 Standard Deviation 56.5313399806
Score Average -0.000865668112954 Standard Deviation 9.49254602213e-05
Elapsed time: 00:00:06

Function Griewank (expected results: mean~2659.06 std_dev~1889.61)
Run	Iteration	Score
==============================
0	3262	-0.000979284359967
5	490	-0.000861284215723
10	119	-0.000899727034571
15	959	-0.000926019754081
20	561	-0.000623119575922
25	1086	-0.0009269255707
30	5000	-0.00739604033412
35	5000	-0.00985728460782
40	73	-0.000988975198918
45	1237	-0.000961233330718

Iterations Average 1943.08 Standard Deviation 1711.96856093
Score Average -0.00181192626743 Standard Deviation 0.002419343976
Elapsed time: 00:02:46

>>> The Full Parallel Bees Algorithm will now be tested <<<
Function Ackley (expected results: mean~128.82 std_dev~29.77)
Run	Iteration	Score
==============================
0	249	-0.000815824993901
5	245	-0.000991993464627
10	262	-0.000923865884382
15	245	-0.000711918205991
20	260	-0.000894062229979
25	239	-0.000838934861065
30	232	-0.000762270598934
35	248	-0.000837784661244
40	253	-0.000834487460807
45	253	-0.000759120719359

Iterations Average 247.26 Standard Deviation 10.366889601
Score Average -0.000808933804627 Standard Deviation 0.000116944882008
Elapsed time: 00:00:14

Function Griewank (expected results: mean~2659.06 std_dev~1889.61)
Run	Iteration	Score
==============================
0	1378	-0.000773109399661
5	2140	-0.000964932912454
10	2141	-0.000939575011903
15	1859	-0.000825160522858
20	536	-0.000966984752333
25	716	-0.00084184791028
30	755	-0.000937815887279
35	764	-0.000784589342995
40	424	-0.000966647675236
45	1826	-0.000948194375068

Iterations Average 1631.74 Standard Deviation 843.486166099
Score Average -0.0008856203907 Standard Deviation 9.85332219911e-05
Elapsed time: 00:01:46
```
On these functions, the partial version provides only marginal benefits in terms of completion time, in comparison with the iterative version.
In contrast, the full version yield around 40% of performance increase in the Griewank function.
On the other hand, it shows worse performances on the Ackley function.
This may be explained by the fact that the full version of parallelism doesn't use elite sites, performing less checks in the hypothesis space per iteration compared to the other versions.
A fine tuning of the *nb* and *nrb* parameters can therefore improve the performances of this version.
## Test function *test_parallel_testing*
In order to test the performance of the parallelism at testing level, it is possible to run the following command:
```sh
$ python bees_algorithm_parallel_testing.py
```
which will perform 50 runs in parallel (on 8 processes) of the Bees Algorithm, producing the following report:
```
Test of the parallel Bees Algorithm on the Ackley function... done.
Iterations 5-values summary (103, 112, 118, 125, 355)
Scores 5-values summary (-0.0009932727283907816, -0.000927073389057842, -0.0008375309658394947, -0.0007589894491997207, -0.00041774252835269365)
Elapsed time: 00:00:02

Test of the parallel Bees Algorithm on the Griewank function... done.
Iterations 5-values summary (64, 846, 1835, 4158, 5000)
Scores 5-values summary (-0.009857284607821315, -0.000986500006021851, -0.0009203912109552181, -0.0008473290407732259, -0.0006019585251350046)
Elapsed time: 00:00:51
```
The resulting elapsed time is roughly 3 times lower than the iterative version.
## Test function *test_visualisation*
This function will show the Bees Algorithm execution on the Griewank target function using an interactive 3D plot. This function is intended to show the potential of the step-by-step visualisation feature.
# References

- [1]: Luca Baronti, Marco Castellani, and Duc Truong Pham. "[An Analysis of the Search Mechanisms of the Bees Algorithm.](
https://www.sciencedirect.com/science/article/abs/pii/S2210650220303990)" Swarm and Evolutionary Computation 59 (2020): 100746.
- [2]: Pham, Duc Truong, et al. "[The Bees Algorithm—A Novel Tool for Complex Optimisation Problems.](https://s3.amazonaws.com/academia.edu.documents/37269572/Pham06_-_The_Bee_Algorithm.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1550167392&Signature=%2BAoKD8QFeUIXvpEjojuAxWu2y0k%3D&response-content-disposition=inline%3B%20filename%3DThe_Bees_Algorithm_-_A_Novel_Tool_for_Co.pdf)" Intelligent production machines and systems. 2006. 454-459.
- [3]: Pham, Duc Truong, and Marco Castellani. "[The bees algorithm: modelling foraging behaviour to solve continuous optimization problems.](https://www.researchgate.net/profile/Marco_Castellani2/publication/229698041_The_Bees_Algorithm_Modelling_foraging_behaviour_to_solve_continuous_optimization_problems/links/0912f50107f349b7de000000/The-Bees-Algorithm-Modelling-foraging-behaviour-to-solve-continuous-optimization-problems.pdf)" Proceedings of the Institution of Mechanical Engineers, Part C: Journal of Mechanical Engineering Science 223.12 (2009): 2919-2938.
- [4]: Pham, Duc Truong, and Marco Castellani. "[Benchmarking and comparison of nature-inspired population-based continuous optimisation algorithms.](https://link.springer.com/article/10.1007/s00500-013-1104-9)" Soft Computing 18.5 (2014): 871-903.
- [5]: Pham, Duc Truong, and Marco Castellani. "[A comparative study of the Bees Algorithm as a tool for function optimisation.](https://www.tandfonline.com/doi/full/10.1080/23311916.2015.1091540)" Cogent Engineering 2.1 (2015): 1091540.

For more references please refer to the README file in the [C++ repository](https://gitlab.com/bees-algorithm/bees_algorithm_cpp).

# Author and License

This library is developed and mantained by Luca Baronti (**gmail** address: *lbaronti*) and released under [GPL v3 license](LICENSE).
