import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from scipy.stats import truncnorm
import pdb
import random

# -------------------------------------------------------------------
# INSTANCE CLASS AND HELPERS-----------------------------------------
# -------------------------------------------------------------------

class graph_search_instance():
    def __init__(self, G, f, r, g):
        self.G = G
        self.G_i = nx.Graph()
        self.G_i.add_node(r)
        self.G_i.add_nodes_from(self.G.neighbors(r))
        self.G_i.add_weighted_edges_from([(r,u,self.G[r][u]["weight"]) for u in self.G.neighbors(r)])
        self.f = f
        self.visited_nodes = [r]
        self.boundary_nodes = set(G.neighbors(r))
        self.root = r
        self.goal = g
        self.cost_to_date = 0
        self.number_of_iterations = 0

    def update_position(self,v):
        # TODO Add tests to check that given node is in fact allowed
        neighbors = set(self.G.neighbors(v))
        self.boundary_nodes = self.boundary_nodes.union(neighbors)-set(self.visited_nodes+[v])
        self.number_of_iterations+=1
        self.cost_to_date += nx.shortest_path_length(self.G_i,self.visited_nodes[-1],v, weight="weight")

        self.visited_nodes.append(v)
        self.G_i.add_nodes_from(neighbors)
        self.G_i.add_weighted_edges_from([(v,u,self.G[v][u]["weight"]) for u in neighbors])
        
        return (neighbors,v==self.goal) #Not sure this boolean condition is the correct one

    # Return the predictions in V_i U partial V_i as a dictionary
    def get_predictions(self):
        return dict((v, self.f[v]) for v in self.boundary_nodes.union(set(self.visited_nodes) ))

    def distance_G_i(self, u, v=None):
        if v is not None:
            return nx.shortest_path_length(self.G_i,source=u, target=v, weight="weight")
        else:
            return nx.shortest_path_length(self.G_i,source=u, weight="weight")

def init_f(G,g,mode=None, epsilon = None, E_1 = None):
    distances = nx.shortest_path_length(G,target=g)
    if mode==None:
        f = {v:distances[v] for v in G.nodes()}
    if mode=='epsilon':
        if epsilon is None:
            epsilon = 0.70
        # set std = eps/2
        # truncate 2 std to the left of 0, and 2 std to the right of 0. 
        eps_v = truncnorm.rvs( a = -2, b = 2,scale = epsilon/2, size=G.number_of_nodes())
        #pdb.set_trace()
        f= {v:(1+eps_v[idx])*distances[v] for idx,v in enumerate(G.nodes())}
    if mode=='additive':
        n = G.number_of_nodes()
        if E_1 is None:
            E_1 = n/2
        errors = np.random.uniform(low = -1, high = 1, size = n)
        errors = (E_1/np.sum(np.abs(errors)))*errors
        f= {v:(distances[v]+errors[idx]) for idx, v in enumerate(G.nodes())}
    if mode=='additive_decremental':
        n = G.number_of_nodes()
        if E_1 is None:
            E_1 = n/2
        errors = np.random.uniform(low = -1, high = 0, size = n)
        errors = (E_1/np.sum(np.abs(errors)))*errors
        f= {v:(distances[v]+errors[idx]) for idx, v in enumerate(G.nodes())}
    return f

# check that predictions satisfy all our intentions
def audit_predictions(f,G,g,mode=None, epsilon = None, E_1 = None):
    distances = nx.shortest_path_length(G,target=g)
    if mode==None:
        return np.all([distances[v]==f[v] for v in G.nodes()])

    if mode=='epsilon':
        if epsilon is None:
            epsilon = 0.70
        # using a list comprehension here raised another weird 'variable is not defined' error. Here's an ugly solution
        correct_f_vals = list()
        for v in G.nodes():
            if v!=g:
                correct_f_vals.append((1-epsilon <= (f[v]/distances[v])) & ((f[v]/distances[v]) <= 1+epsilon ))
            elif v==g:
                correct_f_vals.append(f[v]==0)
        #pdb.set_trace()
        return np.all(correct_f_vals)

    if mode=='additive':
        n = G.number_of_nodes()
        if E_1 is None:
            E_1 = n/2
        errors = list()
        for v in G.nodes():
            errors.append(f[v]-distances[v])
        return np.isclose(np.sum(np.abs(errors)),E_1)

    if mode=='additive_decremental':
        n = G.number_of_nodes()
        if E_1 is None:
            E_1 = n/2
        
        # yet ANOTHER list comprehension thing messing with me
        # errors = [(f[v]-distances[v]) for v, f,  in G.nodes()]
        errors = list()
        for v in G.nodes():
            errors.append(f[v]-distances[v])
        # use np.is_close to avoid False in second condition due to floating pt error
        return np.all([e < 0  for e in errors]) and np.isclose(np.sum(np.abs(errors)),E_1)

# -------------------------------------------------------------------
# EXPLORATION STRATEGIES --------------------------------------------
# -------------------------------------------------------------------

def run_local_search(instance, verbose = False):
    Done = False
    while not Done:
        current_v = instance.visited_nodes[-1]
        current_distances = instance.distance_G_i(current_v)
        
        next_vertex = None
        min_objective = np.inf
        for u in instance.boundary_nodes:
            if (current_distances[u]+instance.f[u]< min_objective):
                next_vertex= u
                min_objective = current_distances[u]+ instance.f[u]
        
        _, Done = instance.update_position(next_vertex)
    if verbose:
        print(f"goal was:{instance.goal} \n visited nodes: {instance.visited_nodes} \n total cost:{instance.cost_to_date} ")
        animate_trajectory(instance)
    return 

# takes 2/3 * d(v_i, u) for small-epsilon case
def run_weighted_local_search(instance, verbose = False):
    Done = False
    while not Done:
        current_v = instance.visited_nodes[-1]
        current_distances = instance.distance_G_i(current_v)
        
        next_vertex = None
        min_objective = np.inf
        for u in instance.boundary_nodes:
            if ((2/3)*current_distances[u]+instance.f[u]< min_objective):
                next_vertex= u
                min_objective = current_distances[u]+ instance.f[u]
        
        _, Done = instance.update_position(next_vertex)
    if verbose:
        print(f"goal was:{instance.goal} \n visited nodes: {instance.visited_nodes} \n total cost:{instance.cost_to_date} ")
        animate_trajectory(instance)
    return 

# trivial searchers: random and greedy
def run_random_search(instance, verbose = False):
    Done = False
    while not Done:
        next_v = random.choice(list(instance.boundary_nodes))
        _, Done = instance.update_position(next_v)
    if verbose:
        print(f"goal was:{instance.goal} \n visited nodes: {instance.visited_nodes} \n total cost:{instance.cost_to_date} ")
        animate_trajectory(instance)
    return

def run_naive_greedy_search(instance, verbose = False):
    Done = False
    while not Done:
        current_v = instance.visited_nodes[-1]
        current_distances = instance.distance_G_i(current_v)
        
        next_vertex = None
        min_objective = np.inf
        for u in instance.boundary_nodes:
            if (instance.f[u]< min_objective):
                next_vertex= u
                min_objective = current_distances[u]+ instance.f[u]
        
        _, Done = instance.update_position(next_vertex)
    if verbose:
        print(f"goal was:{instance.goal} \n visited nodes: {instance.visited_nodes} \n total cost:{instance.cost_to_date} ")
        animate_trajectory(instance)
    return 

# -------------------------------------------------------------------
# PLOTTING METHODS --------------------------------------------------
# -------------------------------------------------------------------
def get_color(x, min = 0, max = 100):
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max)
    cmap = cm.cool
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m.to_rgba(x)

def plot_trajectory(visited_nodes,G):
    labels = dict()
    colors = list()
    for v in G.nodes():
        if v in visited_nodes:
            labels[v] = str(visited_nodes.index(v))
            colors.append(get_color(visited_nodes.index(v), min = 0, max = len(visited_nodes)))
        else:
            labels[v] = 'NA'
            colors.append('#000000ff')
    nx.draw(G, node_color = colors, labels = labels)
    plt.show()
    return

def plot_trajectory_with_predictions(visited_nodes,f,G,r,g, ax = None, pos = None):
    labels = dict()
    colors = list()

    pred_vals = list(f.values())
    max_f = max(pred_vals); min_f = min(pred_vals)
    for v in G.nodes():
        if v in visited_nodes:
            if v==r:
                labels[v]='r'
            elif v==g:
                labels[v]='g'
            else:
                labels[v] = str(visited_nodes.index(v))
            colors.append(get_color(f[v], min = min_f, max = max_f))
        else:
            labels[v] = ''
            colors.append(get_color(f[v], min = min_f, max = max_f))
    if ax is None:
        plt.figure(figsize = (7, 5))
        nx.draw(G, node_color = colors, labels = labels, pos = pos, font_size=  10)
        sm = plt.cm.ScalarMappable(cmap=cm.cool, norm=plt.Normalize(vmin = min_f, vmax=max_f))
        sm._A = []
        plt.colorbar(sm, shrink = 0.9)
        plt.show()
    else:
        nx.draw(G, node_color = colors, labels = labels, pos = pos, ax = ax, font_size=  10)
    return

def animate_trajectory(instance):
    fig, ax = plt.subplots(figsize = (10, 5))
    #og_pos = {v:(0,0) for v in instance.G.nodes()}
    #og_pos[instance.goal] = (1,1)
    #layout = nx.spring_layout(instance.G,  k = 1, pos = og_pos, fixed = [instance.root,instance.goal])
    layout = nx.spring_layout(instance.G)
    visited_nodes = instance.visited_nodes

    n_frames = len(instance.visited_nodes)
    nx.draw(instance.G, pos = layout, ax = ax)

    def update(frame):
            # for each frame, update the data stored on each artist.
            new_labels, new_colors = get_colors(frame)
            nx.draw(instance.G, pos = layout, ax = ax, node_color = new_colors, labels = new_labels)
            return 

    def get_colors(frame):
        colors = list()
        labels = dict()
        for v in instance.G.nodes():
            if v in visited_nodes and visited_nodes.index(v)<=frame:
                labels[v] = str(np.around(instance.f[v], decimals = 1))
                colors.append(get_color(visited_nodes.index(v), min = 0, max = len(visited_nodes)))
            else:
                labels[v] = str(np.around(instance.f[v], decimals = 1))
                if v==instance.goal:
                    colors.append('#fcb900')
                else:
                    colors.append('#abb8c3')
        return labels, colors

    ani = animation.FuncAnimation(fig=fig, func=update, frames=n_frames, interval=50)
    plt.show()

def visualize_predictions(f,G,g,r):
    node_colors = list()
    for v in G.nodes():
        if v==r:
            node_colors.append(get_color(0, min = 0, max = 100))
        elif v==g:
            node_colors.append(get_color(100, min = 0, max = 100))
        else:
            node_colors.append(get_color(50, min = 0, max = 100))
    nx.draw(G, node_color = node_colors, labels = {v:str(np.around(f[v], decimals = 1)) for v in G.nodes()})
    plt.show()

# -------------------------------------------------------------------
# BASIC TESTS -------------------------------------------------------
# -------------------------------------------------------------------

def build_basic_graph(n = 100, type='erdos_renyi'):
    if type=='erdos_renyi':
        connected = False
        while not connected:
            G = nx.erdos_renyi_graph(n, p = 0.1)
            connected = nx.is_connected(G)
        # add basic uniform weights
        weights = {e:1 for e in G.edges()} 

    elif type=='random_tree':
        G = nx.random_tree(n)
        # add basic uniform weights
        weights = {e:1 for e in G.edges()} 

    elif type=='tree':
        G = nx.full_rary_tree(2, 15)
        # add basic uniform weights
        weights = {e:1 for e in G.edges()} 

    for e in G.edges():
        G[e[0]][e[1]]['weight'] = weights[e]
    return G
        
def run_test():
    # Simple test: uniform weights
    G = nx.full_rary_tree(2, 15)
    weights = {e:1 for e in G.edges()}
    for e in G.edges():
        G[e[0]][e[1]]['weight'] = weights[e]

    # identify root and goal
    node_list = list(G.nodes())
    r = node_list[0]
    g = node_list[-1]

    # make predictions
    print('TESTING EPSILON ERROR -------------------------')
    f = init_f(G,g, mode = 'epsilon', epsilon = None)
    print(f"Were predictions constructed correctly? {audit_predictions(f,G,g, mode = 'epsilon', epsilon = None)}")

    # run search and visualize
    tester = graph_search_instance(G, f, r, g)
    run_local_search(tester, verbose = False)
    print(f"I think my search cost me: {tester.cost_to_date}")

    print('TESTING ADDITIVE ERROR -------------------------')
    f = init_f(G,g, mode = 'additive', E_1 = None)
    print(f"Were predictions constructed correctly? {audit_predictions(f,G,g, mode = 'additive', E_1 = None)}")

    # run search and visualize
    tester = graph_search_instance(G, f, r, g)
    run_local_search(tester, verbose = False)
    print(f"I think my search cost me: {tester.cost_to_date}")

    print('TESTING ADDITIVE DECREMENTAL -------------------------')
    f = init_f(G,g, mode = 'additive_decremental', E_1 = None)
    print(f"Were predictions constructed correctly? {audit_predictions(f,G,g, mode = 'additive_decremental', E_1 = None)}")

    # run search and visualize
    tester = graph_search_instance(G, f, r, g)
    run_local_search(tester, verbose = True)
    print(f"I think my search cost me: {tester.cost_to_date}")
