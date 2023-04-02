import sys
import numpy as np
import networkx as nx
import pandas as pd
import math
import matplotlib.pyplot as plt
import datetime


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def create_graph_diagram(graph, names):
    G = nx.DiGraph()
    graph_structure = np.genfromtxt(graph, dtype=str, delimiter=',') # turn graph structure into a numpy array
    if graph_structure.shape == (2,):
        graph_structure = np.array([[graph_structure[0]], [graph_structure[1]]]).T
    G.add_nodes_from(names)
    for row in range(graph_structure.shape[0]):
        G.add_edge(graph_structure[row][0], graph_structure[row][1])
    
    print(G)
    nx.draw(G, with_labels=True, font_weight='bold', pos=nx.shell_layout(G))

    # plot = nx.draw_shell(G)
    # plt.show()

def get_parents(graph, node):
    """
    Find the parents of a particular node given a .gph file

    Kwargs:
        graph: a .gph file containing a previously computed Bayesian network (csv file)
        node: the node name (string)
    
    Returns:
        A list of parent names (numpy array)
    """
    graph_structure = np.genfromtxt(graph, dtype=str, delimiter=',') # turn graph structure into a numpy array
    if graph_structure.shape == (2,):
        graph_structure = np.array([[graph_structure[0]], [graph_structure[1]]]).T
    parent_rows = np.where(graph_structure[:,1] == node)[0]
    parent_list = []
    for row in parent_rows:
        parent_list.append(graph_structure[row][0])
    return np.array(parent_list)

def get_children(graph, node):
    """
    Find the children of a particular node given a .gph file

    Kwargs:
        graph: a .gph file containing a previously computed Bayesian network (csv file)
        node: the node name (string)
    
    Returns:
        A list of child node names (numpy array)
    """
    graph_structure = np.genfromtxt(graph, dtype=str, delimiter=',') # turn graph structure into a numpy array
    if graph_structure.shape == (2,):
        graph_structure = np.array([[graph_structure[0]], [graph_structure[1]]]).T
    child_rows = np.where(graph_structure[1,:] == node)[0]
    child_list = []
    for row in child_rows:
        child_list.append(graph_structure[row][1])
    return np.array(child_list)

def get_parental_instantiations(data, all_nodes, node_parents, node_parent_indices):
    """
    Generate the number of unique instantiation combinations of the parents of a given node

    Kwargs:
        data: raw observed data (numpy array)
        all_parents: a list of all nodes in the data. This is used for indexing purposes (list of strings)
        node_parents: a list of parents for the node (list of strings)
    
    Returns:
        The number of unique parental instantiations (integer)
    """
    qi = 1 # if a node has no parents, then qi = 1
    unique_parent_combinations = None
    if node_parents.shape[0] > 0:
        n_parents = node_parents.shape[0]
        # _, node_indices, _ = np.intersect1d(all_nodes, node_parents, return_indices=True) # get the indices in all_nodes for the parent nodes
        # node_indices = np.flip(np.sort(node_indices))
        parent_data = data[:,node_parent_indices] # pull the columns of parent data for all parent nodes
        parent_max_vals = np.max(parent_data, axis=0)

        possible_parent_vals_list = []
        for val in parent_max_vals:
            possible_parent_vals_list.append(np.arange(1, val + 1))
        possible_parent_vals = np.array(possible_parent_vals_list)

        unique_parent_combinations = np.array(np.meshgrid(*possible_parent_vals)).T.reshape(-1, n_parents) # possible parent vals is an array of arange arrays
        qi = unique_parent_combinations.shape[0]
    return qi, unique_parent_combinations

def compute_score(graph, data, names):
    """
    Compute Bayesian score for a Bayesian network.
    Assume uniform prior such that log(P(G)) = 0.

    Kwargs:
        graph: ".gph" file containing a Bayesian network structure
        data: numpy array containing observed data
        names: numpy array containing the strings for each node's name

    Returns:
        A float (The Bayesian score for the graph given the data)
    """
    # print("Generating score...")
    graph_structure = np.genfromtxt(graph, dtype=str, delimiter=',')
    if graph_structure.shape == (2,):
        graph_structure = np.array([[graph_structure[0]], [graph_structure[1]]]).T
    node_array_list = []
    nodes_parents = [] # a list of np arrays which are lists of parents for each node
    
    # create empty count matrices for each possible node
    for node in names:
        node_idx = np.where(names==node)[0][0]
        node_parent_names = get_parents(graph, node)
        _, node_parent_indices, _ = np.intersect1d(names, node_parent_names, return_indices=True) # get the data column indices for the parent nodes
        node_parent_indices = np.flip(np.sort(node_parent_indices))
        qi, parental_instantiations = get_parental_instantiations(data, names, node_parent_names, node_parent_indices) # get the number of unique instantiations and all parental instantiation combinations
        ri = np.unique(data[:,node_idx]) # number of unique values this node can take on

        node_data = data[:,node_idx] # get the current node's data
        node_data = np.expand_dims(node_data, 1) # expand dims so we can use hstack
        if parental_instantiations is not None:
            parent_data = data[:,node_parent_indices] # get an array containing only parent data
            all_data = np.hstack([parent_data, node_data]) # append current node's data to the last column of the data for the parent nodes
        else:
            all_data = node_data
        count_array = np.zeros([qi, ri.shape[0]])
        for val_idx, val in enumerate(ri):
            if parental_instantiations is not None:
                val_data = all_data[all_data[:,-1]==val] # pick out only the instances where the node takes on a value of val
                val_data = val_data[:,0:-1] # filter out the last column that has current node data so we are matching parental instantiations of parent data
                for idx, combo in enumerate(parental_instantiations):
                    count_array[idx, int(val_idx)-1] = val_data[np.all(val_data==combo, axis=1)].shape[0] # count the number of occurences for a particular combination given that current node value = val and add that to the node's matrix
            else:
                count_array[0, val_idx] = np.sum(all_data == val)
        node_array_list.append(count_array)

    M_array_list = np.array(node_array_list, dtype='object') # this contains the M matrices for each node, which were generated in the above loop. 
    alpha_array_list = []
    for array in M_array_list:
        alpha_array_list.append(np.ones(array.shape))
     
    alpha_array_list = np.array(alpha_array_list)

    m0_sum_vectors = []
    alpha0_sum_vectors = []
    for node_array in node_array_list:
        m0_sum_vectors.append(np.sum(node_array, axis=1))
        alpha0_sum_vectors.append(np.sum(np.ones(node_array.shape), axis=1)) # assuming uniform prior
    m0_sum_vectors = np.array(m0_sum_vectors)
    alpha0_sum_vectors = np.array(alpha0_sum_vectors)
    score = 0
    

    for vec_idx, m0_sum_vector in enumerate(m0_sum_vectors):
        alpha0_sum_vector = alpha0_sum_vectors[vec_idx]
        for idx, val in enumerate(m0_sum_vector):
            # score += np.log(math.gamma(alpha0_sum_vector[idx])/math.gamma(alpha0_sum_vector[idx] + val))
            score += math.lgamma(alpha0_sum_vector[idx]) - math.lgamma(alpha0_sum_vector[idx] + val)

    for idx, M_array in enumerate(M_array_list):
        alpha_array = alpha_array_list[idx]
        for parent in range(M_array.shape[0]):
            for val in range(M_array.shape[1]):
                # score += np.log(math.gamma(alpha_array[parent, val] + M_array[parent, val])/ math.gamma(alpha_array[parent, val]))
                score += math.lgamma(alpha_array[parent, val] + M_array[parent, val]) - math.lgamma(alpha_array[parent, val])

    return score



def generate_network(data, names, output_file):
    """
    Generate a Bayesian network given a set of data

    Kwargs:
        data: numpy array containing observed data
        names: numpy array containing the strings for each node's name
        output_file: a filename string. This is used as the name for the final .gph file that gets outputted
    Returns:
        A .gph file containing the calculated optimal Bayesian network structure
    """
    print("Generating graph structure...")

    # the next two lines just open and close a file with output_file's name to make sure we have an empty file to use
    file = open(output_file, 'w')
    file.close()
    used_names = []

    for node in names:
        current_graph_structure = np.genfromtxt(output_file, dtype=str, delimiter=',')
        if current_graph_structure.shape == (2,):
            current_graph_structure = np.array([[current_graph_structure[0]], [current_graph_structure[1]]]).T
        # TODO try different topological sorts for possible_parents
        possible_parents = names[np.where(names==node)[0][0]+1::] # we can only add parents that come later on in the topological sort
        if current_graph_structure.shape[0] == 0:
            new_parent = np.random.choice(possible_parents) # if we have no prior graph structure, pick a random possible parent node to be a parent
            graph_output_file = open(output_file, 'w')
            graph_output_file.write(new_parent + ',' + node + '\n') # write the new parent,child pair to the graph file
            graph_output_file.close()

        else:
            # current_children = get_children(current_graph_structure, node)
            # current_parents = get_parents(current_graph_structure, node)
            # possible_parents = possible_parents[possible_parents != current_children] # don't add child if it's already a child
            # possible_parents = possible_parents[possible_parents != current_parents] # don't add parent if it's already a parent
            for parent_node in possible_parents:
                open('intermediate.gph', 'w').writelines([l for l in open(output_file).readlines()]) # copy current graph structure to intermediate file
                intermediate_graph = open('intermediate.gph', 'a')
                previous_score = compute_score(output_file, data, names)
                intermediate_graph.write(parent_node + ',' + node + '\n')
                intermediate_graph.close()
                
                new_score = compute_score('intermediate.gph', data, names)
                if new_score > previous_score:
                    open(output_file, 'w').writelines([l for l in open('intermediate.gph').readlines()]) # copy current graph structure to intermediate file
                    
        print("Generated parents for " + node)
        used_names.append(node)
    

def main():
    t0 = datetime.datetime.now()
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    input_data = sys.argv[1]
    output_file = sys.argv[2]
    data = np.genfromtxt(input_data, delimiter=",", skip_header=1) # import data without headers
    names = np.genfromtxt(input_data, delimiter=',', dtype=str, max_rows=1)
    # uncomment below line to switch back to evaluating the example data
    # graph = sys.argv[2]
    generate_network(data, names, output_file)
    score = compute_score(output_file, data, names)
    create_graph_diagram(output_file, names)
    tf = datetime.datetime.now()
    print(tf-t0)
    print(f"Bayesian score for the optimal graph was: {score}")

if __name__ == '__main__':
    main()
