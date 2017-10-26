import numpy as np, os, math
from hive_mem import Tracker, Parameters,Task_Forage
from time import time

parameters = Parameters()  # Create the Parameters class
tracker = Tracker(parameters)  # Initiate tracker
print 'Hive Memory Training with', parameters.num_input, 'inputs,', parameters.num_hnodes, 'hidden_nodes', parameters.num_output, 'outputs and', parameters.output_activation if parameters.output_activation == 'tanh' or parameters.output_activation == 'hardmax' else 'No output activation'

sim_task = Task_Forage(parameters)
if parameters.load_seed: gen_start = int(np.loadtxt(parameters.save_foldername + 'gen_tag'))
else: gen_start = 1
start_t = time()
for gen in range(gen_start, parameters.total_gens):
    best_train_fitness, validation_fitness = sim_task.evolve(gen)
    print 'Gen:', gen, 'Ep_best:', '%.2f' %best_train_fitness, ' Valid_Fit:', '%.2f' %validation_fitness, 'Cumul_valid:', '%.2f'%tracker.hof_avg_fitness, ' out of', '%.2f'%parameters.optimal_score
    # Add best global performance to tracker
    tracker.add_fitness(best_train_fitness, gen) 
    # Add validation global performance to tracker
    tracker.add_hof_fitness(validation_fitness, gen)
end_t = time()
print "Total Runtime: "+str(end_t - start_t)
