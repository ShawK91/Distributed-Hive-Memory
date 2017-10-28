import fastrand
from scipy.special import expit
import numpy as np
cimport numpy as np

cdef np.ndarray[float] fast_hardmax(np.ndarray[float] layer_input):
    return layer_input == np.max(layer_input)

cdef np.ndarray[float] fast_graph_compute(
        np.ndarray[float] input_v,
        np.ndarray[float] memory,
        np.ndarray[float] output,
        np.ndarray[float] w_block_input_bias,
        np.ndarray[float] w_hid_out,
        np.ndarray[float] w_inp,
        np.ndarray[float] w_inpgate,
        np.ndarray[float] w_input_gate_bias,
        np.ndarray[float] w_mem_inpgate,
        np.ndarray[float] w_mem_readgate,
        np.ndarray[float] w_mem_writegate,
        np.ndarray[float] w_readgate,
        np.ndarray[float] w_readgate_bias,
        np.ndarray[float] w_rec_inp,
        np.ndarray[float] w_rec_inpgate,
        np.ndarray[float] w_rec_readgate,
        np.ndarray[float] w_rec_writegate,
        np.ndarray[float] w_writegate,
        np.ndarray[float] w_writegate_bias,
        str output_activation
        ):
    #Input gate
    input_gate_out = expit(np.dot(input_v, w_inpgate) + np.dot(output, w_rec_inpgate) + np.dot(memory, w_mem_inpgate) + w_input_gate_bias)

    #Input processing
    block_input_out = expit(np.dot(input_v, w_inp) + np.dot(output, w_rec_inp) + w_block_input_bias)

    #Gate the Block Input and compute the final input out
    input_out = np.multiply(input_gate_out, block_input_out)

    #Read Gate
    read_gate_out = expit(np.dot(input_v, w_readgate) + np.dot(output, w_rec_readgate) + np.dot(memory, w_mem_readgate) + w_readgate_bias)

    #Compute hidden activation - processing hidden output for this iteration of net run
    hidden_act = np.multiply(read_gate_out, memory) + input_out

    #Write gate (memory cell)
    write_gate_out = expit(np.dot(input_v, w_writegate)+ np.dot(output, w_rec_writegate) + np.dot(memory, w_mem_writegate) + w_writegate_bias)

    #Write to memory Cell - Update memory
    memory += np.multiply(write_gate_out, np.tanh(hidden_act))

    #Compute final output
    output = np.dot(hidden_act, w_hid_out)

    if output_activation == 'tanh':
        output = np.tanh(output)
    elif output_activation == 'hardmax':
        output = fast_hardmax(output)

    return np.array(output).tolist()

