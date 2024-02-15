import numpy as np
import networkx as nx
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import truncnorm
import datetime
import pickle

import pdb
import random
from tqdm import tqdm

from utils import graph_search_instance, init_f, plot_trajectory_with_predictions, build_basic_graph,\
	run_local_search, run_weighted_local_search, run_random_search, run_naive_greedy_search

from modified_a_star import astar_visited

# -------------------------------------------------------------------
# ----- TYPICAL RANDOM ERRORS ON VARRYING GRAPHS --------------------
# -------------------------------------------------------------------

def random_errors_vs_graph_family(n = 100, num_trials = 100, mode="absolute"):
	graphs = ['Random Lobster','Erdos Renyi','Random Tree', 'Circular Ladder']
	# initialize an empty list for each graph family, with keys given by names of graph families
	performance_metrics = {graph:list() for graph in graphs}
	if mode=="absolute":
		errors = [(i+1)*10 for i in range(20)]
		performance_metrics['upper_bound'] = list()
	elif mode=="relative":
		errors = np.linspace(0.01, 0.3, num = 10)

	for error_param in tqdm(errors):
		trial_values = {graph:list() for graph in graphs}
		if mode=="absolute":
			trial_values['upper_bound'] = list()

		for _ in range(num_trials):
			# sample error
			if mode=="absolute":
				error_by_node = build_absolute_errors(n, error_param)
				# compute upperbound
				trial_values['upper_bound'].append(compute_absolute_upperbound(error_by_node))
			elif mode=="relative":
				error_by_node = build_relative_errors(n, error_param)

			for graph in graphs:
				G = build_basic_graph(n, graph)
				# pick goal node g
				node_list = list(G.nodes())
				r, g = random.sample(node_list, k=2)
				if mode=="absolute":
					f = build_predictions_with_absolute_error(G, g, error_by_node)
				elif mode=="relative":
					f = build_predictions_with_relative_error(G, g, error_by_node)
				
				# Set up instance with graph G, goal g, root r, and predictions f
				trial_instance = graph_search_instance( G, f, r, g)
				opt = nx.shortest_path_length(trial_instance.G,source=r, target=g, weight="weight")

				# Run the appropriate search strategy on graph and record performance
				if mode=="absolute":
					run_local_search(trial_instance, verbose = False)
				elif mode=="relative":
					run_weighted_local_search(trial_instance, verbose = False)

				if mode=="absolute":
					trial_values[graph].append(trial_instance.cost_to_date - opt)
				if mode=="relative":
					trial_values[graph].append(trial_instance.cost_to_date/opt)
		for graph in graphs:
			performance_metrics[graph].append( (  np.mean(trial_values[graph])  ,  np.std(trial_values[graph])  ) )
		if mode=="absolute":
			# record upper bound
			performance_metrics['upper_bound'].append( (  np.mean(trial_values['upper_bound'])  ,  np.std(trial_values['upper_bound'])  ) )

	# save results to file
	filename = f'./experiment_results/rand_errors_by_graphtype_experiments_mode={mode}_trials={num_trials}_n={n}_time={datetime.datetime.now()}.pkl'
	with open(filename, 'wb') as fp:
		performance_metrics['n'] = n
		performance_metrics['mode']=mode
		performance_metrics['errors'] = errors
		performance_metrics['num_trials'] = num_trials
		pickle.dump(performance_metrics, fp)
		print(f'dictionary saved successfully to {filename}')

	# plotting
	plt.figure()
	for graph in graphs:
		means, stds = [list(tup) for tup in zip(*performance_metrics[graph])]
		plt.errorbar(errors, means, yerr = stds,marker='o', capsize = 5, label=graph)
		plt.fill_between(errors, np.subtract(means, stds), np.add(means, stds), alpha = 0.5)
	if mode=="absolute":
		plt.ylabel('ALG-OPT')
		plt.xlabel('E_1')
		plt.ylim(bottom=-1)
	elif mode=="relative":
		plt.ylabel('ALG/OPT')
		plt.xlabel('epsilon')
		plt.ylim(bottom=0.9)
	
	plt.legend(loc='upper left')
	plt.show()
	
def build_absolute_errors(n, E1):
	# sample uniformly from the simplex
	errors = np.random.exponential(scale=1, size = n)
	errors = (E1/np.sum(np.abs(errors)))*errors
	return {v:errors[v] for v in range(n)}

def compute_absolute_upperbound(error_dict):
	errors = np.array(list(error_dict.values()))

	n = errors.size
	E_inf_plus = max(errors)
	E_minus = np.sum(errors[errors < 0])
	return E_minus + E_inf_plus*n

def build_predictions_with_absolute_error(G, g, error_dict):
	distances = nx.shortest_path_length(G,target=g)
	return {v:(distances[v]+error_dict[idx]) for idx, v in enumerate(G.nodes())}

def build_relative_errors(n, epsilon):
	eps_v = truncnorm.rvs( a = -2, b = 2,scale = epsilon/2, size=n)
	return {v:eps_v[v] for v in range(n)}

def build_predictions_with_relative_error(G, g, error_dict):
	distances = nx.shortest_path_length(G,target=g)
	return {v:(1+error_dict[idx])*distances[v] for idx,v in enumerate(G.nodes())}

# -------------------------------------------------------------------
# ------------  COMPARISON WITH BASELINES ---------------------------
# -------------------------------------------------------------------

def compare_to_baseline(mode='absolute', num_trials = 2000, verbose = False):
	n = 100
	graph = 'Circular Ladder'

	if mode=="absolute":
		errors = [(i+1)*10 for i in range(20)]
	elif mode=="relative":
		errors = np.linspace(0.01, 0.3, num = 10)

	local_cost_metrics = list()
	naive_greedy_cost_metrics = list()

	for error_param in tqdm(errors):
		local_cost_over_trials = list()
		naive_greedy_cost_over_trials = list()

		for trial in range(num_trials):
			error_by_node = build_absolute_errors(n, error_param)
			G = build_basic_graph(n, graph)
			node_list = list(G.nodes())
			r, g = random.sample(node_list, k=2)
			if mode=="absolute":
				f = build_predictions_with_absolute_error(G, g, error_by_node)
			elif mode=="relative":
				f = build_predictions_with_relative_error(G, g, error_by_node)

			for run_strategy,cost_list in [(run_local_search, local_cost_over_trials),(run_naive_greedy_search, naive_greedy_cost_over_trials)]:
				tester = graph_search_instance( G, f, r, g)
				opt = nx.shortest_path_length(tester.G,source=r, target=g, weight="weight")

				# Execute strategy
				run_strategy(tester, verbose=verbose)
				cost_list.append((tester.cost_to_date-opt)/opt)

		local_cost_metrics.append((np.mean(local_cost_over_trials), np.std(local_cost_over_trials)))
		naive_greedy_cost_metrics.append((np.mean(naive_greedy_cost_over_trials), np.std(naive_greedy_cost_over_trials)))

	# save results to file
	filename = f'./experiment_results/compare_with_greedy_graph={graph}_n={num_trials}_n={n}_time={datetime.datetime.now()}.pkl'
	performance_metrics = {}
	with open(filename, 'wb') as fp:
		performance_metrics['n'] = n
		performance_metrics['graph'] = graph
		performance_metrics['mode']=mode
		performance_metrics['errors'] = errors
		performance_metrics['num_trials'] = num_trials
		performance_metrics['ell_1_greedy'] = local_cost_metrics
		performance_metrics['naive_greedy'] = naive_greedy_cost_metrics
		pickle.dump(performance_metrics, fp)
		print(f'dictionary saved successfully to {filename}')

	plt.figure()
	# Alg 1
	means, stds = [list(tup) for tup in zip(*performance_metrics['ell_1_greedy'])]
	plt.errorbar(performance_metrics['errors'], means, yerr = stds,marker='o', capsize = 5, label='our search')
	plt.fill_between(performance_metrics['errors'], np.subtract(means, stds), np.add(means, stds), alpha = 0.5)

	# naive greedy
	means, stds = [list(tup) for tup in zip(*performance_metrics['naive_greedy'])]
	plt.errorbar(performance_metrics['errors'], means, yerr = stds,marker='o', capsize = 5, label='naive greedy')
	plt.fill_between(performance_metrics['errors'], np.subtract(means, stds), np.add(means, stds), alpha = 0.5)
   
	if mode=='absolute':
		plt.ylabel('(ALG-OPT)/OPT')
		plt.xlabel('E_1')
		plt.ylim(bottom=-1)
	elif mode=="relative":
		plt.ylabel('ALG/OPT')
		plt.xlabel('epsilon')
		#plt.ylim(bottom=0.9)
	
	plt.legend(loc='upper left')
	plt.show()

	return


# -------------------------------------------------------------------
# ------------  VARYING NUMBER OF NODES -----------------------------
# -------------------------------------------------------------------

def changing_n(num_trials = 100, mode="absolute"):
	graphs = ['Random Lobster','Erdos Renyi','Random Tree', 'Circular Ladder']
	# initialize an empty list for each graph family, with keys given by names of graph families
	performance_metrics = {graph:list() for graph in graphs}
	# Relative error setting
	varepsilon=0.2
	n_list = [50, 100, 500, 1000]

	for n in tqdm(n_list):
		trial_values = {graph:list() for graph in graphs}

		for _ in range(num_trials):
			# sample error
			error_by_node = build_relative_errors(n, varepsilon)

			for graph in graphs:
				G = build_basic_graph(n, graph)
				# pick goal node g
				node_list = list(G.nodes())
				r, g = random.sample(node_list, k=2)
				f = build_predictions_with_relative_error(G, g, error_by_node)
				
				# Set up instance with graph G, goal g, root r, and predictions f
				trial_instance = graph_search_instance( G, f, r, g)
				opt = nx.shortest_path_length(trial_instance.G,source=r, target=g, weight="weight")

				# Run the appropriate search strategy on graph and record performance
				run_weighted_local_search(trial_instance, verbose = False)
				trial_values[graph].append(trial_instance.cost_to_date/opt)
		for graph in graphs:
			performance_metrics[graph].append( (  np.mean(trial_values[graph])  ,  np.std(trial_values[graph])  ) )

	# save results to file
	filename = f'./experiment_results/changing_n_mode={mode}_trials={num_trials}_time={datetime.datetime.now()}.pkl'
	with open(filename, 'wb') as fp:
		performance_metrics['n_list'] = n_list
		performance_metrics['mode']='relative'
		performance_metrics['varepsilon'] = varepsilon
		performance_metrics['num_trials'] = num_trials
		pickle.dump(performance_metrics, fp)
		print(f'dictionary saved successfully to {filename}')

	# plotting
	plt.figure()
	for graph in graphs:
		means, stds = [list(tup) for tup in zip(*performance_metrics[graph])]
		plt.errorbar(performance_metrics['n_list'], means, yerr = stds,marker='o', capsize = 5, label=graph)
		plt.fill_between(performance_metrics['n_list'], np.subtract(means, stds), np.add(means, stds), alpha = 0.5)

		plt.ylabel('ALG/OPT')
		plt.xlabel('number of nodes')
		plt.ylim(bottom=0.9)
		ax = plt.gca()
		ax.set_xscale('log')
	
	plt.legend(loc='upper left')
	plt.show()
	return
	 
# -------------------------------------------------------------------
# ------------  PERFORMANCE VS UPPERBOUND TABLE ---------------------
# -------------------------------------------------------------------

def performance_vs_upperbounds_table(n = 100, num_trials = 100):
	graphs = ['Random Lobster','Erdos Renyi','Random Tree', 'Circular Ladder']
	performance_metrics = {graph:list() for graph in graphs}
	performance_metrics['upper_bound'] = list()
	errors = np.linspace(10,2*n, num=20)
	number_of_trials = num_trials

	
	for E1 in tqdm(errors):
		trial_values = {graph:list() for graph in graphs}
		signed_partition = build_absolute_errors(n, E1)
		# compute upperbound
		upperbound_for_sampled_errors = compute_absolute_upperbound(signed_partition)
		performance_metrics['upper_bound'].append(upperbound_for_sampled_errors)
		for _ in range(number_of_trials):
			# randomly shuffle the partition
			signed_errors = list(signed_partition.values())
			# randomly permute in-place
			random.shuffle(signed_errors)
			shuffled_errors = {v:signed_errors[v] for v in range(n)}

			for graph in graphs:
				G = build_basic_graph(n, graph)
				# pick goal node g
				node_list = list(G.nodes())
				r, g = random.sample(node_list, k=2)
				f = build_predictions_with_absolute_error(G, g, shuffled_errors)
				trial_instance = graph_search_instance( G, f, r, g)
				opt = nx.shortest_path_length(trial_instance.G,source=r, target=g, weight="weight")

				# Run search on graph with error and record performance
				run_local_search(trial_instance, verbose = False)
				trial_values[graph].append((trial_instance.cost_to_date - opt)/upperbound_for_sampled_errors)
		for graph in graphs:
			performance_metrics[graph].append( (  np.mean(trial_values[graph])  ,  np.std(trial_values[graph])  ) )

	# save results to file
	filename = f'./experiment_results/perf_vs_upperbounds_table_trials={number_of_trials}_n={n}_time={datetime.datetime.now()}.pkl'
	with open(filename, 'wb') as fp:
		performance_metrics['n'] = n
		performance_metrics['mode']='absolute'
		performance_metrics['errors'] = errors
		performance_metrics['num_trials'] = number_of_trials
		pickle.dump(performance_metrics, fp)
		print(f'dictionary saved successfully to {filename}')
	
	for graph in graphs:
		means, stds = [list(tup) for tup in zip(*performance_metrics[graph])]
		max_mean = max(means)
		std_of_max_mean = stds[means.index(max_mean)]
		print(f'{graph} graph has maximum alg-opt/upperbound = {max_mean} +/- {std_of_max_mean}')
	return

# -------------------------------------------------------------------
# ------- COMPARISON WITH A* SEARCH ---------------------------------
# -------------------------------------------------------------------

# for a set of visited nodes, compute the sum of d(v_i, v_i+1)
# using true distances in G: only valid on trees/hybrid setting
def compute_tour_cost_on_tree(G, visited_nodes):
	cost = 0
	for i in range(len(visited_nodes)-1):
		cost+=nx.shortest_path_length(G,source=visited_nodes[i], target=visited_nodes[i+1], weight="weight")
	return cost

def compare_with_astar(n = 10):
	G = build_basic_graph(n = n, type = 'Random Tree')
	# add basic uniform weights
	weights = {e:1 for e in G.edges()} 
	for e in G.edges():
		G[e[0]][e[1]]['weight'] = weights[e]
	# identify root and goal
	node_list = list(G.nodes())
	r = node_list[0]
	# pick g somewhat removed from r
	g = node_list[-1]
	while nx.shortest_path_length(G,source=r, target=g, weight="weight") < 3:
		g = random.choice(node_list)

	f = init_f(G, g,mode='additive', E_1 = 50)

	# Running A* Search
	# set the heuristic to be the prediction function at v
	h = lambda v, _: f[v] 
	_, astar_visited_nodes = astar_visited(G, source = r, target = g, heuristic = h, weight = "weight")

	# Running our search
	instance = graph_search_instance(G, f, r, g)
	run_local_search(instance, verbose = False)
	our_visited_nodes = instance.visited_nodes

	# compare trajectories
	pos = nx.spring_layout(G)
	_, axs = plt.subplots(1, 2)
	plot_trajectory_with_predictions(astar_visited_nodes,f,G,r,g, ax = axs[0], pos = pos)
	axs[0].set_title(f'A star visited nodes \n Tour cost = {compute_tour_cost_on_tree(G, astar_visited_nodes)}')
	plot_trajectory_with_predictions(our_visited_nodes,f,G,r,g, ax = axs[1], pos = pos)
	axs[1].set_title(f'Our visited nodes \n Tour cost = {instance.cost_to_date}')
	# add colorbar
	sm = plt.cm.ScalarMappable(cmap=cm.cool, norm=plt.Normalize(vmin = min(f.values()), vmax=max(f.values())))
	sm._A = []
	plt.colorbar(sm, shrink = 0.9)
	plt.show()

# -------------------------------------------------------------------
# ------------  PLOTTING FROM FILES ---------------------------------
# -------------------------------------------------------------------

def plot_from_file_random_errors_vs_graphtype(filename):
	file = open(filename,'rb')
	performance_metrics = pickle.load(file)
	file.close()

	graphs = ['Random Lobster','Erdos Renyi','Random Tree', 'Circular Ladder']

	# plotting
	matplotlib.rcParams.update({'font.size': 16})
	plt.figure(figsize = (9,6))
	for graph in graphs:
		means, stds = [list(tup) for tup in zip(*performance_metrics[graph])]
		plt.errorbar(performance_metrics['errors'], means, yerr = stds,marker='o', capsize = 5, label=graph)
		plt.fill_between(performance_metrics['errors'], np.subtract(means, stds), np.add(means, stds), alpha = 0.5)
	if performance_metrics['mode']=="absolute":
		plt.ylabel('ALG-OPT')
		plt.xlabel('E_1')
		plt.ylim(bottom=-1)
	elif performance_metrics['mode']=="relative":
		plt.ylabel('ALG/OPT')
		plt.xlabel('epsilon')
		plt.ylim(bottom=0.9)
	plt.legend(loc='upper left')
	plt.show()

def table_from_file_performance_vs_upperbounds(filename):
	file = open(filename,'rb')
	performance_metrics = pickle.load(file)
	file.close()

	graphs = ['Random Lobster','Erdos Renyi','Random Tree', 'Circular Ladder']

	for graph in graphs:
		means, stds = [list(tup) for tup in zip(*performance_metrics[graph])]
		max_mean = max(means)
		std_of_max_mean = stds[means.index(max_mean)]
		print(f'{graph} graph has maximum alg-opt/upperbound = {max_mean} +/- {std_of_max_mean}')
	return

def plot_from_file_comparison_to_baseline(filename):
	file = open(filename,'rb')
	performance_metrics = pickle.load(file)
	file.close()

	# plotting
	plt.figure()
	# ell_1 greedy search
	means, stds = [list(tup) for tup in zip(*performance_metrics['ell_1_greedy'])]
	plt.errorbar(performance_metrics['errors'], means, yerr = stds,marker='o', capsize = 5, label='our search')
	plt.fill_between(performance_metrics['errors'], np.subtract(means, stds), np.add(means, stds), alpha = 0.5)

	# naive greedy
	means, stds = [list(tup) for tup in zip(*performance_metrics['naive_greedy'])]
	plt.errorbar(performance_metrics['errors'], means, yerr = stds,marker='o', capsize = 5, label='naive greedy')
	plt.fill_between(performance_metrics['errors'], np.subtract(means, stds), np.add(means, stds), alpha = 0.5)
   
	if performance_metrics['mode']=='absolute':
		plt.ylabel('(ALG-OPT)/OPT')
		plt.xlabel('E_1')
		plt.ylim(bottom=-1)
	elif performance_metrics['mode']=="relative":
		plt.ylabel('ALG/OPT')
		plt.xlabel('epsilon')
		#plt.ylim(bottom=0.9)
	
	plt.legend(loc='upper left')
	plt.show()
	return

def plot_from_file_varying_n(filename):
	file = open(filename,'rb')
	performance_metrics = pickle.load(file)
	file.close()

	# plotting
	plt.figure()
	for graph in graphs:
		means, stds = [list(tup) for tup in zip(*performance_metrics[graph])]
		plt.errorbar(performance_metrics['n_list'], means, yerr = stds,marker='o', capsize = 5, label=graph)
		plt.fill_between(performance_metrics['n_list'], np.subtract(means, stds), np.add(means, stds), alpha = 0.5)

		plt.ylabel('ALG/OPT')
		plt.xlabel('number of nodes')
		plt.ylim(bottom=0.9)
		ax = plt.gca()
		ax.set_xscale('log')
	
	plt.legend(loc='upper left')
	plt.show()
	return

# -------------------------------------------------------------------
# ------------  SAMPLE FUNCTION CALLS -------------------------------
# -------------------------------------------------------------------

# # Perform experiment reported in Table 2 (reduced values of n, number of trials, for fast execution)
# performance_vs_upperbounds_table(n = 100, num_trials = 100)

# # Perform experiment reported in Figure 2, Left (reduced number of trials, for fast execution)
# random_errors_vs_graph_family(n = 100, num_trials = 100, mode="absolute")

# # Perform experiment reported in Figure 2, Right (reduced number of trials, for fast execution)
# random_errors_vs_graph_family(n = 100, num_trials = 100, mode="relative")

# # Perform experiment reported in Figure 3
# compare_with_astar(n = 30)

# -------------------------------------------------------------------
# ------------  FIGURE REPRODUCTION ---------------------------------
# -------------------------------------------------------------------

# # Reproduce Table 2 from provided data
# filename='./data_for_figures/table_2.pkl'
# table_from_file_performance_vs_upperbounds(filename)

# # Reproduce Figure 2, Left, from provided data
# filename = './data_for_figures/fig_2_absolute.pkl'
# plot_from_file_random_errors_vs_graphtype(filename)

# # Reproduce Figure 2, Right, from provided data
# filename = './data_for_figures/fig_2_relative.pkl'
# plot_from_file_random_errors_vs_graphtype(filename)

# # Reproduce Figure 3 from provided data
filename = './data_for_figures/fig_3.pkl'
plot_from_file_comparison_to_baseline(filename)



