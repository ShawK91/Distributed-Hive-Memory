import numpy as np, os, math
from hive_mem import Tracker, Parameters,Task_Forage
from time import time

parameters = Parameters()  # Create the Parameters class
if parameters.load_colony:
    gen_start = int(np.loadtxt(parameters.save_foldername + 'gen_tag'))
    tracker = mod.unpickle(parameters.save_foldername + 'tracker')
else:
    tracker = Tracker(parameters, ['best_train', 'valid', 'valid_translated'], '_hive_mem.csv')  # Initiate tracker
    gen_start = 1
#print 'Hive Memory Training with', parameters.num_input, 'inputs,', parameters.num_hnodes, 'hidden_nodes', parameters.num_output, 'outputs and', parameters.output_activation if parameters.output_activation == 'tanh' or parameters.output_activation == 'hardmax' else 'No output activation', 'Exp_opt:', '%.2f'%parameters.expected_optimal, 'Exp_min:', parameters.expected_min,'Exp_max:', parameters.expected_max
sim_task = Task_Forage(parameters)
start_time = time()
for gen in range(gen_start, parameters.total_gens):
    best_train_fitness, validation_fitness = sim_task.evolve(gen, tracker)
    #print 'Gen:', gen, 'Ep_best:', '%.2f' %best_train_fitness, ' Valid_Fit:', '%.2f' %validation_fitness, 'Cumul_valid:', '%.2f'%tracker.all_tracker[1][1], 'translated to', '%.2f'%tracker.all_tracker[2][1], 'out of', '%.2f'%parameters.expected_optimal_translated
    tracker.update([best_train_fitness, validation_fitness, (validation_fitness * (parameters.expected_max-parameters.expected_min)+parameters.expected_min)], gen)
end_time = time()
print  str(end_time - start_time)
