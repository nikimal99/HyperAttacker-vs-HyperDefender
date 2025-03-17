import numpy as np
import networkx as nx
from collections import defaultdict, deque
import random
import networkx as nx
import time
import argparse
import os
from collections import deque, defaultdict
from tqdm import tqdm
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from collections import Counter


def poincare_distance(x:np.ndarray):
    x_norm = np.linalg.norm(x)
    dist = np.arccosh(1 + 2 * (x_norm**2))
    return dist

def load_data():
    adj_mat = np.load('/content/data/amazon-photo/adj.npy', allow_pickle=True)
    embeddings = np.load('/content/data/amazon-photo/embeddings.npy', allow_pickle=True)
    labels = np.load('/content/data/amazon-photo/labels.npy', allow_pickle=True)
    
    print("Number of nodes:{}".format(adj_mat.shape[0]))

    dist = np.zeros(embeddings.shape[0])
    for i in range(embeddings.shape[0]):
        dist[i] = poincare_distance(embeddings[i])

    return adj_mat, embeddings, labels, dist    

def compute_levels(adj_mat, roots):

    num_nodes = adj_mat.shape[0]
    levels = np.full(num_nodes, np.inf)  
    levels[roots] = 0  

    q = deque(roots)
    while q:
        u = q.popleft()
        for v in range(num_nodes):
            if adj_mat[u, v] == 1 and levels[v] == np.inf:
                levels[v] = levels[u] + 1
                q.append(v)
    
    return levels


def swap_nodes(adj_mat, levels, labels=None, num_swaps=10):
    # print("here1")
    num_nodes = adj_mat.shape[0]
    swapped_nodes = set()  

    swaps_performed = 0
    while swaps_performed < num_swaps:
        candidates = []
        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):
                if u in swapped_nodes or v in swapped_nodes:
                    continue
                if levels[u] == levels[v]:  # Same level
                    if np.sum(adj_mat[u]) == np.sum(adj_mat[v]):  # Same degree
                        if labels is None or labels[u] != labels[v]:  # Different labels if provided
                            candidates.append((u, v))
        
        if not candidates:  # No more valid pairs to swap
            break
        # print("here2 in loop u")
      
        u, v = random.choice(candidates)

        adj_mat[[u, v], :] = adj_mat[[v, u], :]
        adj_mat[:, [u, v]] = adj_mat[:, [v, u]]

        if labels is not None:
            labels[u], labels[v] = labels[v], labels[u]

        swapped_nodes.update({u, v})
        swaps_performed += 1
    # print("here2 in loop while")      

    print(f"Performed {swaps_performed} swaps out of requested {num_swaps}")
    return adj_mat, labels if labels is not None else None


def process_hierarchical_graph(adj_mat, poincare_dist, labels, num_swaps):
 
    roots = np.where(poincare_dist == np.min(poincare_dist))[0]
    print("total root nodes and indices of root nodes", len(roots), roots)

    levels = compute_levels(adj_mat, roots)
    print(f"Debug level: {levels}")

    level_counts = Counter(levels)
    # print("Node count at each level:")
    # for level, count in sorted(level_counts.items()):
    #     print(f"Level {int(level)}: {count} nodes")

    perturbed_adj_mat, updated_labels = swap_nodes(adj_mat.copy(), levels, labels.copy(), num_swaps)
    # print("here")
    return perturbed_adj_mat


def hyperbolicity_sample(G, num_samples=100000):
    if len(G.nodes()) < 4:
        raise ValueError("Graph must have at least 4 nodes for hyperbolicity computation.")
    
    curr_time = time.time()
    hyps = []
    for _ in tqdm(range(num_samples)):
        try:
            node_tuple = np.random.choice(G.nodes(), 4, replace=False)
            if not all(nx.has_path(G, node_tuple[i], node_tuple[j]) for i in range(4) for j in range(i + 1, 4)):
                continue  # Skip if any pair of nodes is disconnected
            
            s = []
            d01 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[1], weight=None)
            d23 = nx.shortest_path_length(G, source=node_tuple[2], target=node_tuple[3], weight=None)
            d02 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[2], weight=None)
            d13 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[3], weight=None)
            d03 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[3], weight=None)
            d12 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[2], weight=None)
            
            s.extend([d01 + d23, d02 + d13, d03 + d12])
            s.sort()
            hyps.append((s[-1] - s[-2]) / 2)
        except Exception as e:
            continue  # Skip any errors during sampling

    print('Time for hyp: ', time.time() - curr_time)
    if not hyps:
        print("No valid hyperbolicity values computed.")
        return None, None

    # print("hyps array:", hyps)
    return np.max(hyps), np.mean(hyps)


pert_rates = [0, 5, 10, 15, 20, 25]

if not os.path.exists("/content/strong_hattack_amazonphoto/"):
    os.makedirs("/content/strong_hattack_amazonphoto/")
adj_mat, embeddings, labels, dist = load_data()
num_nodes = dist.shape[0]
new_adj_mat_0 = adj_mat.copy()

for pert_rate in pert_rates:
    num_changes = int((pert_rate / 100.0) * (num_nodes-3000))
    # num_changes = pert_rate
    print("Perturbation rate: {}%".format(pert_rate))
    print("num_changes", num_changes)
    
    for lvl in range(1, 2):
        new_adj_mat_0 = process_hierarchical_graph(new_adj_mat_0, dist, labels, num_changes)
    
    print("Number of edges after attack:{}".format(np.sum(new_adj_mat_0)/2.0))
    print("Difference between the two matrices:{}".format(np.sum(np.abs(adj_mat - new_adj_mat_0))))
    
    adj_sparse = csr_matrix(new_adj_mat_0)
    # print("adj matrices",adj_sparse, new_adj_mat_0)
    graph = nx.from_scipy_sparse_array(adj_sparse)
    
    print('Computing hyperbolicity for graph with', graph.number_of_nodes(), 'nodes and', graph.number_of_edges(), 'edges')
    hyp_max, hyp_mean = hyperbolicity_sample(graph)
    print(f"Perturbation rate: {pert_rate}%")
    print(f"Max hyperbolicity: {hyp_max}, Mean hyperbolicity: {hyp_mean}")

    # plot_degree_distribution(adj_mat, level, labels, "before_attack_{}".format(pert_rate))
    # plot_degree_distribution(new_adj_mat_0, level, labels, "after_attack_{}".format(pert_rate))

    # # plot_graph(original_adjacency_matrix=adj_mat, labels=labels, name="original_{}".format(pert_rate))
    # plot_graph(original_adjacency_matrix=adj_mat, changed_adjacency_matrix=new_adj_mat_0, labels=labels, name="attacked_{}".format(pert_rate))

    np.save(/content/strong_hattack_amazonphoto/adj_ha_{}.npy'.format(pert_rate), new_adj_mat_0)


