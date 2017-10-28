import fastrand
import numpy as np
cimport numpy as np


cdef np.ndarray[float] fast_linear_comb(np.ndarray[float] weight_matrix, np.ndarray[float] layer):
    return np.dot(weight_matrix, layer)

cdef faster_sigmoid()
# TODO faster sigmoid (copy fast sig)

cdef np.ndarray[float] fast_graph_compute(
        np.ndarray[float] input_v,
        np.ndarray[float] memory,
        np.ndarray[float] output,
        np.ndarray[float] w_inpgate,
        np.ndarray[float] w_rec_inpgate,
        np.ndarray[float] w_mem_inpgate,
        np.ndarray[float] w_input_gate_bias,
        np.ndarray[float] w_inp,
        np.ndarray[float] w_rec_inp,
w_block_input_bias

    input = np.mat(input)

    #Input gate
    ig_1 = .linear_combination(input, .w_inpgate)
    ig_2 = .linear_combination(.output, .w_rec_inpgate)
    ig_3 = .linear_combination(memory, .w_mem_inpgate)
    input_gate_out = ig_1 + ig_2 + ig_3 + .w_input_gate_bias
    input_gate_out = .fast_sigmoid(input_gate_out)

    #Input processing
    ig_1 = .linear_combination(input, .w_inp)
    ig_2 = .linear_combination(.output, .w_rec_inp)
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

