import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import datetime
import pickle

import random
import copy
import pdb

from utils import graph_search_instance, init_f, visualize_predictions, plot_trajectory_with_predictions, build_basic_graph,\
    audit_predictions, run_local_search, run_weighted_local_search, run_random_search, run_naive_greedy_search

# -------------------------------------------------------------------
# ----- TYPICAL RANDOM ERRORS ON VARRYING GRAPHS --------------------
# -------------------------------------------------------------------

def random_errors_vs_graph_family(n = 100, mode="absolute"):
    graphs = ['Random Lobster','Erdos Renyi','Random Tree', 'Circular Ladder']
    # initialize an empty list for each graph family, with keys given by names of graph families
    performance_metrics = {graph:list() for graph in graphs}
    if mode=="absolute":
        errors = [(i+1)*10 for i in range(20)]
        performance_metrics['upper_bound'] = list()
    elif mode=="relative":
        errors = np.linspace(0.01, 0.3, num = 10)
    
    number_of_trials = 2000

    for error_param in errors:
        trial_values = {graph:list() for graph in graphs}
        if mode=="absolute":
            trial_values['upper_bound'] = list()

        for _ in range(number_of_trials):
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

                # Run search on graph with error and record performance
                run_local_search(trial_instance, verbose = False)
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
    filename = f'./experiment_results/experiments_trials:{number_of_trials}_n:{n}_time:{datetime.datetime.now()}.pkl'
    with open(filename, 'wb') as fp:
        performance_metrics['n'] = n
        performance_metrics['mode']=mode
        performance_metrics['errors'] = errors
        performance_metrics['num_trials'] = number_of_trials
        pickle.dump(performance_metrics, fp)
        print(f'dictionary saved successfully to {filename}')

    # plotting
    plt.figure()
    for graph in graphs:
        means, stds = [list(tup) for tup in zip(*performance_metrics[graph])]
        #plt.plot(errors, means, linestyle = '-', marker = 'o', label=graph)
        plt.errorbar(errors, means, yerr = stds,marker='o', capsize = 5, label=graph)
        plt.fill_between(errors, np.subtract(means, stds), np.add(means, stds), alpha = 0.5)
    if mode=="absolute":
        # plot upperbound
        #means, stds = [list(tup) for tup in zip(*performance_metrics['upper_bound'])]
        #plt.plot(errors, means, linestyle = '-', marker = 'o', label='upper_bound')
        #plt.fill_between(errors, np.subtract(means, stds), np.add(means, stds), alpha = 0.5)

        #plt.yscale('symlog')
        plt.ylabel('ALG-OPT')
    elif mode=="relative":
        plt.ylabel('ALG/OPT')
    plt.xlabel('Error measure')
    plt.ylim(bottom=-1)
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
# ------------  PERFORMANCE VS UPPERBOUND PLOT ----------------------
# -------------------------------------------------------------------

def performance_vs_upperbounds(n = 100, mode="absolute"):
    graphs = ['Random Lobster','Erdos Renyi','Random Tree', 'Circular Ladder']
    # initialize an empty list for each graph family, with keys given by names of graph families
    performance_metrics = {graph:list() for graph in graphs}
    performance_metrics['upper_bound'] = list()
    errors = np.concatenate((np.linspace(10,50, num=10),np.linspace(50,90, num=15),np.linspace(90,100, num=5)))
    number_of_trials = 1000

    if mode=="absolute":
        for E1 in errors:
            trial_values = {graph:list() for graph in graphs} # Super sketch
            # sample a signed partition
            signed_partition = build_absolute_errors(n, E1)
            # compute upperbound
            performance_metrics['upper_bound'].append(compute_absolute_upperbound(signed_partition))

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
                    trial_values[graph].append(trial_instance.cost_to_date - opt)

            for graph in graphs:
                performance_metrics[graph].append( (  np.mean(trial_values[graph])  ,  np.std(trial_values[graph])  ) )

    # save results to file
    filename = f'./experiment_results/perf_vs_upperbounds_experiments_trials_{number_of_trials}_n_{n}_time_{datetime.datetime.now()}.pkl'
    with open(filename, 'wb') as fp:
        performance_metrics['n'] = n
        performance_metrics['mode']=mode
        performance_metrics['errors'] = errors
        performance_metrics['num_trials'] = number_of_trials
        pickle.dump(performance_metrics, fp)
        print(f'dictionary saved successfully to {filename}')
    
    # get list of upper bounds
    upper_bounds = performance_metrics['upper_bound']

    plt.figure()
    for graph in graphs:#+['upper_bound']:
        means, stds = [list(tup) for tup in zip(*performance_metrics[graph])]
        plt.plot(upper_bounds, means, linestyle='', marker = 'o', label=graph)
        #plt.errorbar(upper_bounds, means, yerr = stds, linestyle='',marker='o', capsize = 5, label=graph)
        plt.yscale('symlog')
        plt.ylabel('ALG-OPT')
    plt.plot(np.linspace(min(upper_bounds),max(upper_bounds)), np.linspace(min(upper_bounds),max(upper_bounds)),'--')
    plt.xlabel('Upper bound')
    plt.ylim(bottom=-1)
    plt.legend()
    plt.show()

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
    G = build_basic_graph(n = n, type = 'random_tree')
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


performance_vs_upperbounds(n = 100, mode="absolute")
# random_errors_vs_graph_family(n = 100, mode="absolute")