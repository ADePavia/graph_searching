import numpy as np
import networkx as nx
from networkx.algorithms import tournament
import matplotlib.pyplot as plt
import random
import copy
import pdb

from utils import graph_search_instance, init_f, visualize_predictions, audit_predictions, run_local_search, run_weighted_local_search, run_random_search, run_naive_greedy_search

# -------------------------------------------------------------------
# ADVERSARIAL PREDICTIONS -------------------------------------------
# -------------------------------------------------------------------

def adversarial_f(G,r,g,mode=None, epsilon = None, E_1 = None):
    distances = nx.shortest_path_length(G,target=g)
    if mode==None:
        raise Exception('Not yet implemented')
    # build uniform ball of predictions around the root, all other predictions epsilon-fluctuations
    if mode=='epsilon':
        if epsilon is None:
            epsilon = 0.70

        # build a base of random epsilon errors
        f= {v:(1+np.random.uniform(low = -epsilon, high = epsilon))*distances[v] for v in G.nodes()}

        # build a uniform ball around root
        OPT = distances[r]
        R_r =epsilon*OPT
        dist_to_root = nx.shortest_path_length(G,target=r)
        for v in G.nodes():
            if dist_to_root[v] <= R_r:
                f[v] = (OPT-R_r)*(1+epsilon)
    # allocate positive error randomly over neighbors of g, all other predictions correct.
    if mode=='additive':
        n = G.number_of_nodes()
        # build a base of true predictions
        f= {v:distances[v] for v in G.nodes()}

        if E_1 is None:
            E_1 = n/2
        
        g_neighbors = list(G.neighbors(g))
        errors = np.random.uniform(low = 0, high = 1, size = len(g_neighbors))
        errors = (E_1/np.sum(np.abs(errors)))*errors
        
        for idx, v in enumerate(g_neighbors):
            f[v] = distances[v] + errors[idx]
    return f

# -------------------------------------------------------------------
# COMPARING STRATEGIES ----------------------------------------------
# -------------------------------------------------------------------

def build_basic_graph(n = 100, type='erdos_renyi'):
    if type=='erdos_renyi':
        connected = False
        while not connected:
            G = nx.erdos_renyi_graph(n, p = 0.1)
            connected = nx.is_connected(G)
    elif type=='tree':
        G = nx.full_rary_tree(2, 15)

    # add basic uniform weights
    weights = {e:1 for e in G.edges()} 
    for e in G.edges():
        G[e[0]][e[1]]['weight'] = weights[e]
    return G
        
def compare_strategies(mode, graph_generator, f_generator, num_trials, f_auditor = None, verbose=False):
    local_cost = list()
    weighted_local_cost = list()
    naive_greedy_cost = list()
    for _ in range(num_trials):
        G = graph_generator()
        
        # identify root and goal
        node_list = list(G.nodes())
        r = node_list[0]
        g = node_list[-1]

        # a "typical" instance shouldn't have r and g connected:
        while (g in G.neighbors(r)) or (g==r):
            g = random.choice(node_list)

        f = f_generator(G, r, g)
        if not (f_auditor is None):
            assert f_auditor(f,G,g)

        for run_strategy,cost_list in [(run_local_search, local_cost), (run_weighted_local_search, weighted_local_cost),
                        (run_naive_greedy_search, naive_greedy_cost)]:
            # initialize a new tester for each of our strategies
            tester = graph_search_instance( G, f, r, g)
            run_strategy(tester, verbose=verbose)
            if mode=='competitive_ratio':
                CR = tester.cost_to_date/nx.shortest_path_length(tester.G,source=r, target=g, weight="weight")
                cost_list.append(CR)
            if mode=='additive_cost':
                additive_cost = tester.cost_to_date - nx.shortest_path_length(tester.G,source=r, target=g, weight="weight")
                cost_list.append(additive_cost)

    if verbose:   
        for strategy_name, cost_list in [('local search', local_cost), ('weighted local search', weighted_local_cost),
                    ('naive greedy search', naive_greedy_cost)]:
            print(f"Strategy {strategy_name} mean CR = {np.mean(cost_list)} +/- {np.std(cost_list)}")
    return [(np.mean(cost_list), np.std(cost_list)) for cost_list in [local_cost, weighted_local_cost, naive_greedy_cost]]

# -------------------------------------------------------------------
# EXPERIMENTS -------------------------------------------------------
# -------------------------------------------------------------------

# For different values of n, constructs adversarial epsilon predictions over Erdos-Renyi random graphs with
# different numbers of nodes (n) but constant p. Demonstrates how naive greedy search incurrs a competitive
# ratio that blows up with n, but our local and weighted search strategies do not.
# Adversarial predictions place a uniform-looking ball around r, and then random epsilon-predictions elsewhere.
def demonstrate_adversarial_epsilon_examples(n_list, num_trials):
    local_stats = list()
    weighted_local_stats = list()
    naive_greedy_stats = list()
    for n in n_list:
        f_generator = lambda G, r, g: adversarial_f(G,r,g,mode='epsilon', epsilon = 0.3)
        graph_generator = lambda: build_basic_graph(n = n)
        local_tuple, weighted_tuple, naive_greedy_tuple = compare_strategies('competitive_ratio', graph_generator, 
                                                    f_generator, num_trials, verbose=False)

        local_stats.append(local_tuple)
        weighted_local_stats.append(weighted_tuple)
        naive_greedy_stats.append(naive_greedy_tuple)

    plt.figure()
    for strategy_name, stat_list in [('local search', local_stats), ('weighted local search', weighted_local_stats),
                    ('naive greedy search', naive_greedy_stats)]:
        means, stds = [list(tup) for tup in zip(*stat_list)]
        #plt.plot(np.log(n_list), np.log(means), label=strategy_name)
        #plt.fill_between(np.log(n_list), np.subtract(means, stds), np.add(means, stds), alpha = 0.5)
        plt.plot(np.log(n_list), np.log(means), linestyle = '-', marker = 'o', label=strategy_name)
    plt.xlabel('log(number of nodes)')
    plt.ylabel('log(Competitive ratio)')
    plt.legend()
    plt.show()

    plt.figure()
    for strategy_name, stat_list in [('local search', local_stats), ('weighted local search', weighted_local_stats),
                    ('naive greedy search', naive_greedy_stats)]:
        means, stds = [list(tup) for tup in zip(*stat_list)]
        plt.plot(np.log(n_list), means, linestyle = '-', marker = 'o',label=strategy_name)
        plt.fill_between(np.log(n_list), np.subtract(means, stds), np.add(means, stds), alpha = 0.5)
    plt.xlabel('log(number of nodes)')
    plt.ylabel('Competitive ratio')
    plt.legend()
    plt.show()
        
# constructs two subgraphs, one containing r and one containing g, with minimum vertex cut
# k between r and g. To ensure that all graphs in the experiment have comparable number
# of edges: adds total_added_edges edgeweight to the graph overall. Only
# k of those span the cliques containing r and g. At the end, we check to make sure the minimum
# vertex cut is of size k.  
# Right now, subgraphs are Erdos-Renyi (n/2, p) for p = 0.2
# G has edges of unit length. When we start adding edges, we randomly pick edges not yet in
# the graph, without replacement.
def construct_clique_based_k_bottleneck(n, total_added_edges, k, verbose = False):
    assert total_added_edges >= k
    min_r_g_cut = np.inf

    while min_r_g_cut!=k:
        # r_subgraph = nx.complete_graph(int(n/2))
        # g_subgraph = nx.complete_graph(int(n/2))
        p = 0.3
        r_subgraph = nx.erdos_renyi_graph(int(n/2), p)
        while not nx.is_connected(r_subgraph):
            r_subgraph = nx.erdos_renyi_graph(int(n/2), p)
        g_subgraph = nx.erdos_renyi_graph(int(n/2), p)
        while not nx.is_connected(g_subgraph):
            g_subgraph = nx.erdos_renyi_graph(int(n/2), p)

        G = nx.disjoint_union(r_subgraph, g_subgraph)
        # basic uniform weights
        for e in G.edges():
            G[e[0]][e[1]]['weight'] = 1

        nodelist = list(G.nodes())
        # we want to pick r and g to be in different cliques
        r = nodelist[0]
        g = nodelist[-1]
        # if somehow the merge indexing/relabeling doesn't put the two cliques at either end of the nodelist,
        while nx.has_path(G, r, g):
            # randomly choose a different node until we get one in the other clique
            g = random.choice(nodelist)

        # build k random bridges
        # get lists of nodes which are the connected components in G. We expect two
        cc_list = [list(c) for c in nx.connected_components(G)]
        assert len(cc_list)==2
        for _ in range(k):
            # pick two random nodes until we get some we haven't seen before
            v_1 = random.choice(cc_list[0])
            v_2 = random.choice(cc_list[1])
            while G.has_edge(v_1, v_2):
                v_1 = random.choice(cc_list[0])
                v_2 = random.choice(cc_list[1])
            G.add_edge(v_1, v_2, weight  = 1)
            
        # add the remaining edgeweight
        for _ in range(total_added_edges - k):
            # randomly pick a subgraph
            cc = random.choice(cc_list)
            # randomly pick two nodes (without replacement) from that subgraph
            v_1, v_2 = random.sample(cc, 2)
            if G.has_edge(v_1, v_2):
                G[e[0]][e[1]]['weight'] *= 0.5
            else:
                G.add_edge(v_1, v_2, weight  = 1)
        # compute minumum r-g vertex cut
        vertex_cut = nx.minimum_node_cut(G, r, g)
        min_r_g_cut = len(vertex_cut)
    
    if verbose:
        # dummy predictions for visualization
        f = {v:0 for v in G.nodes()}
        visualize_predictions(f,G,g,r)
        plt.show()
    return G

# compute minumum s-t cut assuming flow capacity = 1 on all edges
def compute_mincut_unit_capacity(G, s, t):
    # we have to provide a 'capacity' attribute. To do so without
    # editing G, create a deepcopy.
    deepcopy_G = copy.deepcopy(G)
    for e in G.edges():
        deepcopy_G[e[0]][e[1]]['capacity'] = 1
    return nx.minimum_cut_value(deepcopy_G, s, t, capacity = 'capacity')

# explores the effect on adding a greater number of paths from r to g
def demonstrate_improvement_with_greater_connectivity(n, k_list, num_trials, titlestring = None):
    local_stats = list()
    weighted_local_stats = list()
    naive_greedy_stats = list()

    total_added_edges = max(k_list)
    for k in k_list:
        f_generator = lambda G, r, g: init_f(G,g,mode='epsilon', epsilon = None)
        graph_generator = lambda: construct_clique_based_k_bottleneck(n = n, total_added_edges= total_added_edges, k = k)
        local_tuple, weighted_tuple, naive_greedy_tuple = compare_strategies('competitive_ratio', graph_generator, 
                                                                    f_generator, num_trials, verbose=False)
        local_stats.append(local_tuple)
        weighted_local_stats.append(weighted_tuple)
        naive_greedy_stats.append(naive_greedy_tuple)

    plt.figure()
    for strategy_name, stat_list in [('local search', local_stats), ('weighted local search', weighted_local_stats),
                    ('naive greedy search', naive_greedy_stats)]:
        means, stds = [list(tup) for tup in zip(*stat_list)]
        plt.plot(k_list, means, linestyle = '-', marker = 'o', label=strategy_name)
        plt.fill_between(k_list, np.subtract(means, stds), np.add(means, stds), alpha = 0.5)
        #plt.plot(np.log(n_list), np.log(means), linestyle = '-', marker = 'o', label=strategy_name)
    plt.xlabel('Value of minimum r-g cut')
    plt.ylabel('Competitive ratio')
    if not (titlestring is None):
        plt.title(titlestring)
    plt.legend()
    plt.show()

def demonstrate_random_performance_vs_predicted_UB_additive_decremental(n, num_trials, E_1_list, titlestring = None):
    local_stats = list()
    for E_1 in E_1_list:
        graph_generator = lambda: build_basic_graph(n = n)
        f_generator = lambda G, r, g: init_f(G,g,mode='additive_decremental', E_1 = E_1)
        local_tuple, _, _ = compare_strategies('additive_cost', graph_generator, f_generator, num_trials, verbose=False)
        local_stats.append(local_tuple)
    
    plt.figure()
    for strategy_name, stat_list in [('local search', local_stats)]:
        means, stds = [list(tup) for tup in zip(*stat_list)]
        plt.plot(E_1_list, means, linestyle = '-', marker = 'o', label='Algorithmic performance under random errors')
        plt.fill_between(E_1_list, np.subtract(means, stds), np.add(means, stds), alpha = 0.5)
    plt.plot(E_1_list, E_1_list, linestyle = '--', marker = 'o', label='Worst-case upperbound')
    plt.yscale('symlog')
    plt.ylabel('ALG - OPT (symmetrical logarithmic scale)')
    plt.xlabel('E1 Error')
    if not (titlestring is None):
        plt.title(titlestring)
    plt.legend()
    plt.show()

def demonstrate_random_vs_adversarial_error_additive_E1(n, num_trials, E_1_list, titlestring = None):
    random_local_stats = list()
    adversarial_local_stats = list()
    for E_1 in E_1_list:
        graph_generator = lambda: build_basic_graph(n = n)
        random_f_generator = lambda G, r, g: init_f(G,g,mode='additive', E_1 = E_1)
        adversarial_f_generator = lambda G, r, g: adversarial_f(G, r, g, mode = 'additive', E_1 = E_1)

        for stat_list, f_generator in [(random_local_stats, random_f_generator), (adversarial_local_stats, adversarial_f_generator)]:
            local_tuple, _, _ = compare_strategies('additive_cost', graph_generator, f_generator, num_trials, verbose=False)
            stat_list.append(local_tuple)
    
    plt.figure()
    for f_type, stat_list in [('random E1 errors', random_local_stats), ('challenging E1 errors', adversarial_local_stats)]:
        means, stds = [list(tup) for tup in zip(*stat_list)]
        plt.plot(E_1_list, means, linestyle = '-', marker = 'o', label=f_type)
        plt.fill_between(E_1_list, np.subtract(means, stds), np.add(means, stds), alpha = 0.5)
    #plt.yscale('symlog')
    plt.ylabel('ALG - OPT')
    plt.xlabel('E1 Error')
    if not (titlestring is None):
        plt.title(titlestring)
    plt.legend()
    plt.show()


# num_trials = 200
# rand_vs_adversarial_titlestring = f'Impact of random versus \'challening\' E1 additive error \n {num_trials} trials with 100-node, p = 0.1 Erdos-Renyi random graphs'    
# demonstrate_random_vs_adversarial_error_additive_E1(n = 100, num_trials = num_trials, E_1_list = [0, 5, 10, 25, 50],
#                                                                  titlestring = rand_vs_adversarial_titlestring)


# G = build_basic_graph(n = 10, type = 'erdos_renyi')
# nodelist = list(G.nodes())
# r = nodelist[0]
# g = nodelist[-1]
# f = adversarial_f(G, r, g, mode = 'additive',  E_1 = None)

# print(f"Were predictions created correctly? {audit_predictions(f,G,g,mode='additive', E_1 = None)}")

# visualize_predictions(f,G,g,r)
# plt.show()

#construct_clique_based_k_bottleneck(n = 20, total_added_edges = 10, k = 5, verbose = True)

connectivity_titlestring = 'Impact of small r-g vertex cuts on performance \n 100 trials on 100-node graphs'
demonstrate_improvement_with_greater_connectivity(n = 100, k_list = [1, 3, 5, 10, 15], num_trials = 100, 
                                                titlestring = connectivity_titlestring)

# worstcase_gap_titlestring = 'Comparing worst-case bounds versus actual performance \n Random admissible E1 error for varying values of E1 \n 1000 trials with 100-node, p = 0.1 Erdos-Renyi random graphs'
# demonstrate_random_performance_vs_predicted_UB_additive_decremental(n = 100, num_trials = 1000, E_1_list =[5, 10, 50, 90, 99], 
#                                              titlestring = worstcase_gap_titlestring)


#demonstrate_adversarial_epsilon_examples(n_list = [50, 100, 200], num_trials = 10)