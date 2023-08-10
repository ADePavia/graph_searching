import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import pdb
import random


class graph_search_instance():
    def __init__(self, G, f, r, g):
        self.G = G
        self.G_i = nx.Graph()
        self.G_i.add_node(r)
        self.G_i.add_nodes_from(self.G.neighbors(r))
        self.G_i.add_weighted_edges_from([(r,u,G[r][u]["weight"]) for u in self.G.neighbors(r)])
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
        self.G_i.add_weighted_edges_from([(v,u,G[v][u]["weight"]) for u in neighbors])
        
        return (neighbors,v==self.goal) #Not sure this boolean condition is the correct one

    # Return the predictions in V_i U partial V_i as a dictionary
    def get_predictions(self):
        return dict((v, self.f[v]) for v in self.boundary_nodes.union(set(self.visited_nodes) ))

    def distance_G_i(self, u, v=None):
        if v is not None:
            return nx.shortest_path_length(self.G_i,source=u, target=v, weight="weight")
        else:
            return nx.shortest_path_length(self.G_i,source=u, weight="weight")

def run_random_search(instance):
    Done = False
    while not Done:
        next_v = random.choice(list(instance.boundary_nodes))
        _, Done = instance.update_position(next_v)

    print(f"goal was:{instance.goal} \n visited nodes: {instance.visited_nodes} \n total cost:{instance.cost_to_date} ")

def run_local_search(instance):
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
    print(f"goal was:{instance.goal} \n visited nodes: {instance.visited_nodes} \n total cost:{instance.cost_to_date} ")
    animate_trajectory(instance)
    return 

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

def run_animation_example():
    fig, ax = plt.subplots()
    t = np.linspace(0, 3, 40)
    g = -9.81
    v0 = 12
    z = g * t**2 / 2 + v0 * t

    v02 = 5
    z2 = g * t**2 / 2 + v02 * t

    scat = ax.scatter(t[0], z[0], c="b", s=5, label=f'v0 = {v0} m/s')
    line2 = ax.plot(t[0], z2[0], label=f'v0 = {v02} m/s')[0]
    ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='Z [m]')
    ax.legend()


    def update(frame):
        # for each frame, update the data stored on each artist.
        x = t[:frame]
        y = z[:frame]
        # update the scatter plot:
        data = np.stack([x, y]).T
        scat.set_offsets(data)
        # update the line plot:
        line2.set_xdata(t[:frame])
        line2.set_ydata(z2[:frame])
        return (scat, line2)


    ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)
    plt.show()

def animate_trajectory(instance):
    fig, ax = plt.subplots(figsize = (10, 5))
    layout = nx.spring_layout(instance.G)
    visited_nodes = instance.visited_nodes

    n_frames = len(instance.visited_nodes)
    nx.draw(G, pos = layout, ax = ax)

    def update(frame):
            # for each frame, update the data stored on each artist.
            new_labels, new_colors = get_colors(frame)
            nx.draw(G, pos = layout, ax = ax, node_color = new_colors, labels = new_labels)
            return 

    def get_colors(frame):
        colors = list()
        labels = dict()
        for v in G.nodes():
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

def init_f(G,g,mode=None, epsilon = None):
    distances = nx.shortest_path_length(G,target=g)
    if mode==None:
        f = {v:distances[v] for v in G.nodes()}
    if mode=='epsilon':
        if epsilon is None:
            epsilon = 0.70
        f= {v:(1+np.random.uniform(low = -epsilon, high = epsilon))*distances[v] for v in G.nodes()}
    return f

# build weighted graph
n = 5
#G = nx.barbell_graph(15, 5)
G = nx.random_intern_as_graph(10000)
weights = {e:1 for e in G.edges()} #max([e[0],e[1]]) for e in G.edges()}
for e in G.edges():
    G[e[0]][e[1]]['weight'] = weights[e]

# identify root and goal
node_list = list(G.nodes())
r = node_list[0]
g = node_list[-1]

# make predictions
f = init_f(G,g, mode = 'epsilon')

tester = graph_search_instance( G, f, r, g)
run_local_search(tester)
