import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sys
import datetime


def get_data(input_data):
    """
    Import data from a .csv file for processing

    Kwargs:
        file: a .csv file
    
    Returns:
        Two numpy arrays: data (contains data without headers) and names (names from data columns)
    """
    data = np.genfromtxt(input_data, delimiter=",", skip_header=1) # import data without headers
    names = np.genfromtxt(input_data, delimiter=',', dtype=str, max_rows=1)
    return data, names


def q_learn(model, n_iterations):
    """
    Use Q-Learning to learn the Q-function for an MDP

    This is a model-free approach. It iterates to learn the optimal value for actions given particular states (in other words, it tries to create an optimal q function).

    Kwargs:
        data: Batched MDP observations data
        names: Names for each column from observations
        gamma: Future reward discount factor
        alpha: Learning rate

    Returns:
        A numpy array with dimensions (n_states) x (n_states). It is a Q-function that maps states to actions. 
    """
    data, col_names = get_data('data/' + model.name + '.csv')
    gamma = model.gamma
    alpha = model.learning_rate
    n_states = model.n_states
    n_actions = model.n_actions
    action_array = np.arange(1,n_actions+1)
    name = model.name
    q_array = np.zeros((n_states, n_actions+1)) # initialize empty array. Will store value of each action associated with each state
    optimal_q = np.zeros((n_states, 2)) # optimal q maps each state to an action
    #TODO Calculate q_array here
    sorted_data = data[data[:,0].argsort()] # create array sorted in order of state number
    
    # Because gamma = 1 for medium, using a set number of iterations instead of value based convergence
    for i in range(n_iterations):
        print(f"iteration {i}")
        for obs_idx, observation in enumerate(data):
            # iterate through observations and update q_array each time
            state = int(observation[0])
            action = int(observation[1])
            reward = observation[2]
            next_state = int(observation[3])

            # update q_array value for each observation
            q_array[state-1, action] += alpha * (reward + gamma * np.max(q_array[next_state-1, :]) - q_array[state-1, action])

    
    
    best_vals = np.max(q_array[:,1::], axis=1)
    for row_idx, row in enumerate(q_array):
        # Pick action corresponding to best value. The random.choice picks a random action out of the best ones if there is a tie
        best_val_idx = np.random.choice(np.where(q_array[row_idx, :] == best_vals[row_idx])[0]) 
        
        # update optimal state-action mapping in the optimal q matrix
        optimal_q[row_idx][0] = row[0]
        optimal_q[row_idx][1] = best_val_idx
    
    # missing data choice 1: choose random action
    # optimal_q[:,1][optimal_q[:,1]==0] = np.random.choice(action_array) 

    # missing data choice 2: Use action from previous state
    missing_data = np.where(optimal_q[:,1]==0)[0]
    for state_idx in missing_data:
        if state_idx > 0:
            optimal_q[state_idx][1] = optimal_q[state_idx-1][1]
        else:
            optimal_q[:,state_idx] = np.random.choice(action_array)

    write_to_txt(optimal_q, model.name + '.policy')
    

def write_to_txt(optimal_q, filename):
    """
    Writes the optimal state-action pair calculated by q-learning to a text file
    """
    q_output_file = open(filename, 'w')
    for row in optimal_q:
        # state = row[0]
        action = row[1]
        # q_output_file.write(state + ',' + action + '\n') # write the state, action pair to the text file
        q_output_file.write(f'{action}' + '\n') # The ith line contains the optimal action for the ith state (per Ed post)

    q_output_file.close()


class model:
    """
    Contains gamma, alpha, n_states, n_actions for a given model.
    These values are pulled from the AA228 Project 2 Page
    """
    def __init__(self, learning_rate, gamma, n_states, n_actions, name):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_states = n_states
        self.n_actions = n_actions
        self.name = name

    
def main():
    t0 = datetime.datetime.now()
    # Below models are using parameters from the Project 2 Page on the AA228 Website
    small_learning_model = model(0.01, 0.95, 100, 4, 'small')
    medium_learning_model = model(0.01, 1, 50000, 7, 'medium')
    large_learning_model = model(0.01, 0.95, 312020, 9, 'large')
    #TODO Note to self: might need to use if statements in q_learn or make different q_learn functions depending on which csv we're using
    # small_q_matrix = q_learn(small_learning_model, 300)
    # medium_q_matrix = q_learn(medium_learning_model, 300)
    large_q_matrix = q_learn(large_learning_model, 300)
    
    # pick the action for each state that gave the highest value
    tf = datetime.datetime.now()
    print(tf-t0)
  

if __name__ == '__main__':
    #TODO Possible optimizations to implement later...
    # Not all states are equally important (per project info). Maybe weight by number of times we see something 
    main()

