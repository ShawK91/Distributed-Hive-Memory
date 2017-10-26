from random import randint
import fastrand
import math
import  cPickle
import random
import numpy as np
from scipy.special import expit


class Drone_Default:
    def __init__(self, params, drone_id, num_input, num_hnodes, num_mem, num_output):
        self_drone_id = drone_id; self.params = params
        self.num_input = num_input; self.num_output = num_output; self.num_hnodes = num_hnodes

        #Mean and std
        mean = 0; std_input = 1.0/(math.sqrt(num_input)); std_hnodes = 1.0/(math.sqrt(num_hnodes)); std_mem = 1.0/(math.sqrt(num_mem)); std_output = 1.0/(math.sqrt(num_output))

        #Input gate
        self.w_inpgate = np.mat(np.random.normal(mean, std_input, (num_input, num_hnodes)))
        self.w_rec_inpgate = np.mat(np.random.normal(mean, std_output, (num_output, num_hnodes)))
        self.w_mem_inpgate = np.mat(np.random.normal(mean, std_mem, (num_mem, num_hnodes)))

        #Block Input
        self.w_inp = np.mat(np.random.normal(mean, std_input, (num_input, num_hnodes)))
        self.w_rec_inp = np.mat(np.random.normal(mean, std_output, (num_output, num_hnodes)))

        #Forget gate
        self.w_readgate = np.mat(np.random.normal(mean, std_input, (num_input, num_hnodes)))
        self.w_rec_readgate = np.mat(np.random.normal(mean, std_output, (num_output, num_hnodes)))
        self.w_mem_readgate = np.mat(np.random.normal(mean, std_mem, (num_mem, num_hnodes)))

        #Memory write gate
        self.w_writegate = np.mat(np.random.normal(mean, std_input, (num_input, num_hnodes)))
        self.w_rec_writegate = np.mat(np.random.normal(mean, std_output, (num_output, num_hnodes)))
        self.w_mem_writegate = np.mat(np.random.normal(mean, std_mem, (num_mem, num_hnodes)))

        #Output weights
        self.w_hid_out= np.mat(np.random.normal(mean, std_hnodes, (num_hnodes, num_output)))

        #Biases
        self.w_input_gate_bias = np.mat(np.random.normal(mean, 0.1, (1, num_hnodes)))
        self.w_block_input_bias = np.mat(np.random.normal(mean, 0.1, (1, num_hnodes)))
        self.w_readgate_bias = np.mat(np.random.normal(mean, 0.1, (1, num_mem)))
        self.w_writegate_bias = np.mat(np.random.normal(mean, 0.1, (1, num_mem)))

        #Adaptive components (plastic with network running)
        self.output = np.mat(np.zeros((1, self.num_output)))

        self.param_dict = {'w_inpgate': self.w_inpgate,
                           'w_rec_inpgate': self.w_rec_inpgate,
                           'w_mem_inpgate': self.w_mem_inpgate,
                           'w_inp': self.w_inp,
                           'w_rec_inp': self.w_rec_inp,
                            'w_readgate': self.w_readgate,
                            'w_rec_readgate': self.w_rec_readgate,
                            'w_mem_readgate': self.w_mem_readgate,
                            'w_writegate': self.w_writegate,
                            'w_rec_writegate': self.w_rec_writegate,
                            'w_mem_writegate': self.w_mem_writegate,
                           'w_hid_out': self.w_hid_out,
                            'w_input_gate_bias': self.w_input_gate_bias,
                           'w_block_input_bias': self.w_block_input_bias,
                            'w_readgate_bias': self.w_readgate_bias,
                           'w_writegate_bias': self.w_writegate_bias}

    def hardmax(self, layer_input):
        return layer_input == np.max(layer_input)

    def expanded_graph_compute(self, input, memory): #Feedforwards the input and computes the forward pass of the network
        input = np.mat(input)

        #Input gate
        ig_1 = self.linear_combination(input, self.w_inpgate)
        ig_2 = self.linear_combination(self.output, self.w_rec_inpgate)
        ig_3 = self.linear_combination(memory, self.w_mem_inpgate)
        input_gate_out = ig_1 + ig_2 + ig_3 + self.w_input_gate_bias
        input_gate_out = self.fast_sigmoid(input_gate_out)

        #Input processing
        ig_1 = self.linear_combination(input, self.w_inp)
        ig_2 = self.linear_combination(self.output, self.w_rec_inp)
        block_input_out = ig_1 + ig_2 + self.w_block_input_bias
        block_input_out = self.fast_sigmoid(block_input_out)

        #Gate the Block Input and compute the final input out
        input_out = np.multiply(input_gate_out, block_input_out)

        #Read Gate
        ig_1 = self.linear_combination(input, self.w_readgate)
        ig_2 = self.linear_combination(self.output, self.w_rec_readgate)
        ig_3 = self.linear_combination(memory, self.w_mem_readgate)
        read_gate_out = ig_1 + ig_2 + ig_3 + self.w_readgate_bias
        read_gate_out = self.fast_sigmoid(read_gate_out)

        #Memory Output
        memory_output = np.multiply(read_gate_out, memory)

        #Compute hidden activation - processing hidden output for this iteration of net run
        hidden_act = memory_output + input_out

        #Write gate (memory cell)
        ig_1 = self.linear_combination(input, self.w_writegate)
        ig_2 = self.linear_combination(self.output, self.w_rec_writegate)
        ig_3 = self.linear_combination(memory, self.w_mem_writegate)
        write_gate_out = ig_1 + ig_2 + ig_3 + self.w_writegate_bias
        write_gate_out = self.fast_sigmoid(write_gate_out)

        #Write to memory Cell - Update memory
        memory += np.multiply(write_gate_out, np.tanh(hidden_act))

        #Compute final output
        self.output = self.linear_combination(hidden_act, self.w_hid_out)
        if self.params.output_activation == 'tanh': self.output = np.tanh(self.output)
        elif self.params.output_activation == 'hardmax': self.output = self.hardmax(self.output)

        return np.array(self.output).tolist(), memory

    def graph_compute(self, input, memory): #Feedforwards the input and computes the forward pass of the network
        input = np.mat(input)

        #Input gate
        input_gate_out = expit(np.dot(input, self.w_inpgate) + np.dot(self.output, self.w_rec_inpgate) + np.dot(memory, self.w_mem_inpgate) + self.w_input_gate_bias)

        #Input processing
        block_input_out = expit(np.dot(input, self.w_inp) + np.dot(self.output, self.w_rec_inp) + self.w_block_input_bias)

        #Gate the Block Input and compute the final input out
        input_out = np.multiply(input_gate_out, block_input_out)

        #Read Gate
        read_gate_out = expit(np.dot(input, self.w_readgate) + np.dot(self.output, self.w_rec_readgate) + np.dot(memory, self.w_mem_readgate) + self.w_readgate_bias)

        #Compute hidden activation - processing hidden output for this iteration of net run
        hidden_act = np.multiply(read_gate_out, memory) + input_out

        #Write gate (memory cell)
        write_gate_out = expit(np.dot(input, self.w_writegate)+ np.dot(self.output, self.w_rec_writegate) + np.dot(memory, self.w_mem_writegate) + self.w_writegate_bias)

        #Write to memory Cell - Update memory
        memory += np.multiply(write_gate_out, np.tanh(hidden_act))

        #Compute final output
        self.output = np.dot(hidden_act, self.w_hid_out)
        if self.params.output_activation == 'tanh': self.output = np.tanh(self.output)
        elif self.params.output_activation == 'hardmax': self.output = self.hardmax(self.output)

        return np.array(self.output).tolist()

    def reset(self):
        self.output = np.mat(np.zeros((1,self.num_output)))

class Drone_Detached:
    def __init__(self, params, drone_id, num_input, num_hnodes, num_mem, num_output):
        self_drone_id = drone_id; self.params = params
        self.num_input = num_input; self.num_output = num_output; self.num_hnodes = num_hnodes; self.num_mem = num_mem

        #Mean and std
        mean = 0; std_input = 1.0/(math.sqrt(num_input)); std_hnodes = 1.0/(math.sqrt(num_hnodes)); std_mem = 1.0/(math.sqrt(num_mem)); std_output = 1.0/(math.sqrt(num_output))

        #Input gate
        self.w_inpgate = np.mat(np.random.normal(mean, std_input, (num_input, num_hnodes)))
        self.w_rec_inpgate = np.mat(np.random.normal(mean, std_output, (num_output, num_hnodes)))
        self.w_mem_inpgate = np.mat(np.random.normal(mean, std_mem, (num_mem, num_hnodes)))

        #Block Input
        self.w_inp = np.mat(np.random.normal(mean, std_input, (num_input, num_hnodes)))
        self.w_rec_inp = np.mat(np.random.normal(mean, std_output, (num_output, num_hnodes)))

        #Read gate
        self.w_readgate = np.mat(np.random.normal(mean, std_input, (num_input, num_mem)))
        self.w_rec_readgate = np.mat(np.random.normal(mean, std_output, (num_output, num_mem)))
        self.w_mem_readgate = np.mat(np.random.normal(mean, std_mem, (num_mem, num_mem)))
        self.w_mem_hid = np.mat(np.random.normal(mean, std_mem, (num_mem, num_hnodes)))

        #Memory write gate
        self.w_writegate = np.mat(np.random.normal(mean, std_input, (num_input, num_mem)))
        self.w_rec_writegate = np.mat(np.random.normal(mean, std_output, (num_output, num_mem)))
        self.w_mem_writegate = np.mat(np.random.normal(mean, std_mem, (num_mem, num_mem)))
        self.w_hid_mem = np.mat(np.random.normal(mean, std_hnodes, (num_hnodes, num_mem)))

        #Output weights
        self.w_hid_out= np.mat(np.random.normal(mean, std_hnodes, (num_hnodes, num_output)))

        #Biases
        self.w_input_gate_bias = np.mat(np.random.normal(mean, 0.1, (1, num_hnodes)))
        self.w_block_input_bias = np.mat(np.random.normal(mean, 0.1, (1, num_hnodes)))
        self.w_readgate_bias = np.mat(np.random.normal(mean, 0.1, (1, num_mem)))
        self.w_writegate_bias = np.mat(np.random.normal(mean, 0.1, (1, num_mem)))

        #Adaptive components (plastic with network running)
        self.output = np.mat(np.zeros((1, self.num_output)))

        self.param_dict = {'w_inpgate': self.w_inpgate,
                           'w_rec_inpgate': self.w_rec_inpgate,
                           'w_mem_inpgate': self.w_mem_inpgate,
                           'w_inp': self.w_inp,
                           'w_rec_inp': self.w_rec_inp,
                            'w_readgate': self.w_readgate,
                            'w_rec_readgate': self.w_rec_readgate,
                            'w_mem_readgate': self.w_mem_readgate,
                            'w_writegate': self.w_writegate,
                            'w_rec_writegate': self.w_rec_writegate,
                            'w_mem_writegate': self.w_mem_writegate,
                           'w_hid_out': self.w_hid_out,
                            'w_input_gate_bias': self.w_input_gate_bias,
                           'w_block_input_bias': self.w_block_input_bias,
                            'w_readgate_bias': self.w_readgate_bias,
                           'w_writegate_bias': self.w_writegate_bias,
                           'w_mem_hid': self.w_mem_hid,
                           'w_hid_mem': self.w_hid_mem}

    def hardmax(self, layer_input):
        return layer_input == np.max(layer_input)

    def expanded_graph_compute(self, input, memory): #Feedforwards the input and computes the forward pass of the network
        input = np.mat(input)

        #Input gate
        ig_1 = self.linear_combination(input, self.w_inpgate)
        ig_2 = self.linear_combination(self.output, self.w_rec_inpgate)
        ig_3 = self.linear_combination(memory, self.w_mem_inpgate)
        input_gate_out = ig_1 + ig_2 + ig_3 + self.w_input_gate_bias
        input_gate_out = self.fast_sigmoid(input_gate_out)

        #Input processing
        ig_1 = self.linear_combination(input, self.w_inp)
        ig_2 = self.linear_combination(self.output, self.w_rec_inp)
        block_input_out = ig_1 + ig_2 + self.w_block_input_bias
        block_input_out = np.tanh(block_input_out)

        #Gate the Block Input and compute the final input out
        input_out = np.multiply(input_gate_out, block_input_out)

        #Read Gate
        ig_1 = self.linear_combination(input, self.w_readgate)
        ig_2 = self.linear_combination(self.output, self.w_rec_readgate)
        ig_3 = self.linear_combination(memory, self.w_mem_readgate)
        read_gate_out = ig_1 + ig_2 + ig_3 + self.w_readgate_bias
        read_gate_out = self.fast_sigmoid(read_gate_out)

        #Memory Output
        memory_output = np.multiply(read_gate_out, memory)

        #Compute hidden activation - processing hidden output for this iteration of net run
        hidden_act = np.tanh(self.linear_combination(memory_output, self.w_mem_hid)) + input_out

        #Write gate (memory cell)
        ig_1 = self.linear_combination(input, self.w_writegate)
        ig_2 = self.linear_combination(self.output, self.w_rec_writegate)
        ig_3 = self.linear_combination(memory, self.w_mem_writegate)
        write_gate_out = ig_1 + ig_2 + ig_3 + self.w_writegate_bias
        write_gate_out = self.fast_sigmoid(write_gate_out)

        #Write to memory Cell - Update memory
        memory += np.multiply(write_gate_out, np.tanh(self.linear_combination(hidden_act, self.w_hid_mem)))

        #Compute final output
        self.output = self.linear_combination(hidden_act, self.w_hid_out)
        if self.params.output_activation == 'tanh': self.output = np.tanh(self.output)
        elif self.params.output_activation == 'hardmax': self.output = self.hardmax(self.output)


        return np.array(self.output).tolist(), memory

    def graph_compute(self, input, memory): #Feedforwards the input and computes the forward pass of the network
        input = np.mat(input)

        #Input gate
        input_gate_out = expit(np.dot(input, self.w_inpgate) + np.dot(self.output, self.w_rec_inpgate) + np.dot(memory, self.w_mem_inpgate) + self.w_input_gate_bias)

        #Input processing
        block_input_out = np.tanh(np.dot(input, self.w_inp) + np.dot(self.output, self.w_rec_inp) + self.w_block_input_bias)

        #Gate the Block Input and compute the final input out
        input_out = np.multiply(input_gate_out, block_input_out)

        #Read Gate
        read_gate_out = expit(np.dot(input, self.w_readgate) + np.dot(self.output, self.w_rec_readgate) + np.dot(memory, self.w_mem_readgate) + self.w_readgate_bias)

        #Memory Output
        memory_output = np.multiply(read_gate_out, memory)

        #Compute hidden activation - processing hidden output for this iteration of net run
        hidden_act = np.tanh(np.dot(memory_output, self.w_mem_hid)) + input_out

        #Write gate (memory cell)
        write_gate_out = expit(np.dot(input, self.w_writegate) + np.dot(self.output, self.w_rec_writegate) + np.dot(memory, self.w_mem_writegate) + self.w_writegate_bias)

        #Write to memory Cell - Update memory
        memory += np.multiply(write_gate_out, np.tanh(np.dot(hidden_act, self.w_hid_mem)))

        #Compute final output
        self.output = np.dot(hidden_act, self.w_hid_out)
        if self.params.output_activation == 'tanh': self.output = np.tanh(self.output)
        elif self.params.output_activation == 'hardmax': self.output = self.hardmax(self.output)


        return np.array(self.output).tolist(), memory

    def reset(self):
        self.output = np.mat(np.zeros((1,self.num_output)))

class Drone_FF:
    def __init__(self, params, drone_id, num_input, num_hnodes, num_mem, num_output):
        self_drone_id = drone_id; self.params = params
        self.num_input = num_input; self.num_output = num_output; self.num_hnodes = num_hnodes

        # Mean and std
        mean = 0; std_input = 1.0 / (math.sqrt(num_input)); std_hnodes = 1.0 / (math.sqrt(num_hnodes));

        #Block Input
        self.w_inp = np.mat(np.random.normal(mean, std_input, (num_input, num_hnodes)))

        #Output weights
        self.w_hid_out= np.mat(np.random.normal(mean, std_hnodes, (num_hnodes, num_output)))

        #Biases
        self.w_inp_bias = np.mat(np.random.normal(mean, 0.1, (1, num_hnodes)))
        self.w_hid_out_bias = np.mat(np.random.normal(mean, 0.1, (1, num_output)))

        self.param_dict = {'w_inp': self.w_inp,
                           'w_hid_out': self.w_hid_out,
                           'w_inp_bias': self.w_inp_bias,
                           'w_hid_out_bias': self.w_hid_out_bias}


    def linear_combination(self, w_matrix, layer_input): #Linear combine weights with inputs
        return np.dot(w_matrix, layer_input) #Linear combination of weights and inputs

    def relu(self, layer_input):    #Relu transformation function
        for x in range(len(layer_input)):
            if layer_input[x] < 0:
                layer_input[x] = 0
        return layer_input

    def fast_sigmoid(self, layer_input): #Sigmoid transform
        layer_input = expit(layer_input)
        return layer_input

    def softmax(self, layer_input): #Softmax transform
        layer_input = np.exp(layer_input)
        layer_input = layer_input / np.sum(layer_input)
        return layer_input

    def hardmax(self, layer_input):
        return layer_input == np.max(layer_input)

    def graph_compute(self, input, _): #Feedforwards the input and computes the forward pass of the network
        input = np.mat(input)

        #Input processing
        hidden_act = self.fast_sigmoid(self.linear_combination(input, self.w_inp) + self.w_inp_bias)

        #Compute final output
        self.output = self.linear_combination(hidden_act, self.w_hid_out) + self.w_hid_out_bias
        if self.params.output_activation == 'tanh': self.output = np.tanh(self.output)
        elif self.params.output_activation == 'hardmax': self.output = self.hardmax(self.output)

        return np.array(self.output).tolist()

    def reset(self):
        return



class Hive:
    def __init__(self, params, mean = 0, std = 1):
        self.params = params;

        #Hive Memory
        self.memory = np.mat(np.zeros((1, self.params.num_mem)))

        #Initialize drones (controllers)
        self.all_drones = []
        for drone_id in range(self.params.num_drones):
            if params.grumb_topology == 1:
                self.all_drones.append(
                    Drone_Default(params, drone_id, params.num_input, params.num_hnodes, params.num_mem, params.num_output))
            if params.grumb_topology == 2:
                self.all_drones.append(
                    Drone_Detached(params, drone_id, params.num_input, params.num_hnodes, params.num_mem, params.num_output))
            if params.grumb_topology == 3:
                self.all_drones.append(
                    Drone_FF(params, drone_id, params.num_input, params.num_hnodes, params.num_mem, params.num_output))

    def reset(self):
        self.memory = np.mat(np.zeros((1,self.params.num_mem)))
        for drone in self.all_drones:
            drone.reset()

    def forward(self, input, drone_id):
        output = self.all_drones[drone_id].graph_compute(input, self.memory)
        return output[0]

class Fast_SSNE:
    def __init__(self, parameters):
        self.current_gen = 0
        self.parameters = parameters;
        self.population_size = self.parameters.population_size;
        self.num_elitists = int(self.parameters.elite_fraction * parameters.population_size)
        if self.num_elitists < 1: self.num_elitists = 1
        self.num_input = self.parameters.num_input;
        self.num_hidden = self.parameters.num_hnodes;
        self.num_output = self.parameters.num_output

    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[fastrand.pcg32bounded(len(offsprings))])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def regularize_weight(self, weight):
        if weight > self.parameters.weight_magnitude_limit:
            weight = self.parameters.weight_magnitude_limit
        if weight < -self.parameters.weight_magnitude_limit:
            weight = -self.parameters.weight_magnitude_limit
        return weight

    def crossover_inplace(self, hive_1, hive_2):
        for drone_1, drone_2 in zip(hive_1.all_drones, hive_2.all_drones):

            keys = list(drone_1.param_dict.keys())

            # References to the variable tensors
            W1 = drone_1.param_dict
            W2 = drone_2.param_dict
            num_variables = len(W1)
            if num_variables != len(W2): print 'Warning: Genes for crossover might be incompatible'

            # Crossover opertation [Indexed by column, not rows]
            num_cross_overs = fastrand.pcg32bounded(num_variables * 2)  # Lower bounded on full swaps
            for i in range(num_cross_overs):
                tensor_choice = fastrand.pcg32bounded(num_variables)  # Choose which tensor to perturb
                receiver_choice = random.random()  # Choose which gene to receive the perturbation
                if receiver_choice < 0.5:
                    ind_cr = fastrand.pcg32bounded(W1[keys[tensor_choice]].shape[-1])  #
                    W1[keys[tensor_choice]][:, ind_cr] = W2[keys[tensor_choice]][:, ind_cr]
                    #W1[keys[tensor_choice]][ind_cr, :] = W2[keys[tensor_choice]][ind_cr, :]
                else:
                    ind_cr = fastrand.pcg32bounded(W2[keys[tensor_choice]].shape[-1])  #
                    W2[keys[tensor_choice]][:, ind_cr] = W1[keys[tensor_choice]][:, ind_cr]
                    #W2[keys[tensor_choice]][ind_cr, :] = W1[keys[tensor_choice]][ind_cr, :]

    def hive_crossover(self, hive_1, hive_2): #Transfer drone between two hives
        num_transfer = fastrand.pcg32bounded(self.parameters.num_drones)+1
        for _ in range(num_transfer):
            source_id = fastrand.pcg32bounded(self.parameters.num_drones)
            receiver_id = fastrand.pcg32bounded(self.parameters.num_drones)
            if fastrand.pcg32bounded(2) == 0:
                for key in hive_1.all_drones[receiver_id].param_dict.keys():
                    hive_2.all_drones[receiver_id].param_dict[key][:] = hive_1.all_drones[source_id].param_dict[key]
            else:
                for key in hive_2.all_drones[receiver_id].param_dict.keys():
                    hive_1.all_drones[receiver_id].param_dict[key][:] = hive_2.all_drones[source_id].param_dict[key]

    def homogenize(self, hive):
        alpha_drone_index = random.choice([i for i in range(len(hive.all_drones))])
        for drone_id, drone in enumerate(hive.all_drones):
            if drone_id != alpha_drone_index:
                if random.random() < 0.5:
                    for key in hive.all_drones[alpha_drone_index].param_dict.keys():
                        drone.param_dict[key][:] = hive.all_drones[alpha_drone_index].param_dict[key]

    def homogenize_gates(self, hive):
        alpha_drone_index = random.choice([i for i in range(len(hive.all_drones))])
        for drone_id, drone in enumerate(hive.all_drones):
            if drone_id != alpha_drone_index:
                gate_keys = ['w_inpgate', 'w_rec_inpgate', 'w_mem_inpgate', 'w_readgate', 'w_rec_readgate', 'w_mem_readgate', 'w_writegate', 'w_rec_writegate', 'w_mem_writegate', 'w_input_gate_bias', 'w_readgate_bias', 'w_writegate_bias']
                for key in gate_keys:
                    drone.param_dict[key][:] = hive.all_drones[alpha_drone_index].param_dict[key]

    def mutate_inplace(self, hive):
        mut_strength = 0.1
        num_mutation_frac = 0.1
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.05


        for drone in hive.all_drones:
            # References to the variable keys
            keys = list(drone.param_dict.keys())
            W = drone.param_dict
            num_structures = len(keys)
            ssne_probabilities = np.random.uniform(0,1,num_structures)*2


            for ssne_prob, key in zip(ssne_probabilities, keys): #For each structure
                if random.random()<ssne_prob:

                    num_mutations = fastrand.pcg32bounded(int(math.ceil(num_mutation_frac * W[key].size)))  # Number of mutation instances
                    for _ in range(num_mutations):
                        ind_dim1 = fastrand.pcg32bounded(W[key].shape[0])
                        ind_dim2 = fastrand.pcg32bounded(W[key].shape[-1])
                        random_num = random.random()

                        if random_num < super_mut_prob:  # Super Mutation probability
                            W[key][ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength *
                                                                                          W[key][
                                                                                              ind_dim1, ind_dim2])
                        elif random_num < reset_prob:  # Reset probability
                            W[key][ind_dim1, ind_dim2] = random.gauss(0, 1)

                        else:  # mutauion even normal
                            W[key][ind_dim1, ind_dim2] += random.gauss(0, mut_strength *W[key][
                                                                                              ind_dim1, ind_dim2])

                        # Regularization hard limit
                        W[key][ind_dim1, ind_dim2] = self.regularize_weight(
                            W[key][ind_dim1, ind_dim2])

    def copy_individual(self, master, replacee):  # Replace the replacee individual with master
        for master_drone, replacee_drone in zip(master.all_drones, replacee.all_drones):
            keys = master_drone.param_dict.keys()
            for key in keys:
                replacee_drone.param_dict[key][:] = master_drone.param_dict[key]

    def reset_genome(self, gene):
        for drone in gene.all_drones:
            keys = drone.param_dict
            for key in keys:
                dim = drone.param_dict[key].shape
                drone.param_dict[key][:] = np.mat(np.random.uniform(-1, 1, (dim[0], dim[1])))

    def epoch(self, all_hives, fitness_evals):

        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        index_rank = self.list_argsort(fitness_evals); index_rank.reverse()
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)

        #Extinction step (Resets all the offsprings genes; preserves the elitists)
        if random.random() < self.parameters.extinction_prob: #An extinction event
            print
            print "######################Extinction Event Triggered#######################"
            print
            for i in offsprings:
                if random.random() < self.parameters.extinction_magnituide and not (i in elitist_index):  # Extinction probabilities
                    self.reset_genome(all_hives[i])

        # Figure out unselected candidates
        unselects = []; new_elitists = []
        for i in range(self.population_size):
            if i in offsprings or i in elitist_index:
                continue
            else:
                unselects.append(i)
        random.shuffle(unselects)

        # Elitism step, assigning elite candidates to some unselects
        for i in elitist_index:
            replacee = unselects.pop(0)
            new_elitists.append(replacee)
            self.copy_individual(master=all_hives[i], replacee=all_hives[replacee])

        # Crossover for unselected genes with 100 percent probability
        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            unselects.append(unselects[fastrand.pcg32bounded(len(unselects))])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(new_elitists);
            off_j = random.choice(offsprings)
            self.copy_individual(master=all_hives[off_i], replacee=all_hives[i])
            self.copy_individual(master=all_hives[off_j], replacee=all_hives[j])
            self.crossover_inplace(all_hives[i], all_hives[j])

        # Crossover for selected offsprings
        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if random.random() < self.parameters.crossover_prob: self.crossover_inplace(all_hives[i], all_hives[j])
            if random.random() < self.parameters.hive_crossover_prob: self.hive_crossover(all_hives[i], all_hives[j])

        # Mutate all genes in the population except the new elitists plus homozenize
        for i in range(self.population_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.parameters.homogenize_prob: self.homogenize(all_hives[i])
                if random.random() < self.parameters.homogenize_gates_prob: self.homogenize_gates(all_hives[i])
                if random.random() < self.parameters.mutation_prob: self.mutate_inplace(all_hives[i])


def unpickle(filename):
    import pickle
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def pickle_object(obj, filename):
    with open(filename, 'wb') as output:
        cPickle.dump(obj, output, -1)

