import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt 
from itertools import combinations


def swap(x,i,j):
    x_new = x.copy()
    x_new[i], x_new[j] = x[j], x[i]
    return x_new 

def cal_dist(x,G):
    # 1 2 3 4 5 , 1-2, 2-3, 3-4, 4-5, 5-1 
    return np.sum([G[x[i]][x[i+1]]['weight'] for i in range(len(x)-1)] ) + \
           G[x[-1]][x[0]]['weight']

def tabu_tsp(G, x0, n_tabu=10, n=5, max_iter=10000):
    x0 = np.array(x0)
    n_nodes = len(x0)
    best_x = x0 
    best_f = cal_dist(x0,G)
    curr_x = x0 
    tabu_list = [] 
    n_iter = 0 
    possible_change = list(combinations(range(1, n_nodes),2))

    while n_iter < max_iter : 
        n_iter +=1 
        if len(tabu_list) > n_tabu:
            tabu_list  = tabu_list[1:]
        cand_change = np.random.choice(range(len(possible_change)), size=n, replace=False)
        cand_x = np.array([swap(curr_x, *possible_change[i]) for i in cand_change])
        cand_fval = np.apply_along_axis(cal_dist,1,cand_x,G)
        min_ind = np.argmin(cand_fval)

        if not '.'.join(cand_x[min_ind]) in tabu_list:
            curr_x = cand_x[min_ind]
            tabu_list.append('.'.join(cand_x[min_ind]))
        
            if cand_fval[min_ind] < best_f : 
                best_f = cand_fval[min_ind]
                best_x = cand_x[min_ind]
                print(n_iter, best_x, best_f)
    return best_x, best_f 



def main():
    n_nodes = 5
    G = nx.complete_graph(np.arange(n_nodes).astype('str'))
    G.add_weighted_edges_from((u,v,np.random.randint(1,3)) for u,v in G.edges())
    G.nodes()
    G.edges()
    G.edges(data=True)
    edges = G.edges()
    edges_label = dict(((u,v), str(G[u][v]['weight'])) for u,v in edges)
    figure = plt.figure()
    pos = nx.spring_layout(G)
    nx.draw(G, pos = pos)
    nx.draw_networkx_labels(G, pos=pos, edges_label=edges_label)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edges_label)
    x0 = np.random.permutation(np.arange(n_nodes).astype('str'))
    best_x = tabu_tsp(G,x0, n_tabu=5)
    plt.show()
if __name__ == "__main__":
    main()