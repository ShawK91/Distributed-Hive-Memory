import numpy as np, os, math
import mod_hive_mem as mod, sys
from random import randint

class Tracker(): #Tracker
    def __init__(self, parameters, vars_string, project_string):
        self.vars_string = vars_string; self.project_string = project_string
        self.foldername = parameters.save_foldername
        self.all_tracker = [[[],0.0,[]] for _ in vars_string] #[Id of var tracked][fitnesses, avg_fitness, csv_fitnesses]
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)

    def update(self, updates, generation):
        for update, var in zip(updates, self.all_tracker):
            var[0].append(update)

        #Constrain size of convolution
        if len(self.all_tracker[0][0]) > 100: #Assume all variable are updated uniformly
            for var in self.all_tracker:
                var[0].pop(0)

        #Update new average
        for var in self.all_tracker:
            var[1] = sum(var[0])/float(len(var[0]))

        if generation % 10 == 0:  # Save to csv file
            for i, var in enumerate(self.all_tracker):
                var[2].append(np.array([generation, var[1]]))
                filename = self.foldername + self.vars_string[i] + self.project_string
                np.savetxt(filename, np.array(var[2]), fmt='%.3f', delimiter=',')

class Parameters:
    def __init__(self):
        self.population_size = 100
        self.load_colony = 0
        self.total_gens = 50
        self.is_hive_mem = True #Is Hive memory connected/active? If not, no communication between the agents
        self.num_evals = 5 #Number of different maps to run each individual before getting a fitness

        #NN specifics
        self.num_hnodes = 25
        self.num_mem = 10
        self.grumb_topology = 1 #1: Default (hidden nodes cardinality attached to that of mem (No trascriber))
                                #2: Detached (Memory independent from hidden nodes (transcribing function))
                                #3: FF (Normal Feed-Forward Net)
        self.output_activation = 'tanh' #tanh or hardmax

        #SSNE stuff
        self.elite_fraction = 0.03
        self.crossover_prob = 0.05
        self.mutation_prob = 0.9
        self.homogenize_prob = 0.005
        self.homogenize_gates_prob = 0.01
        self.hive_crossover_prob = 0.03
        self.extinction_prob = 0.004 #Probability of extinction event
        self.extinction_magnituide = 0.5 #Probabilty of extinction for each genome, given an extinction event
        self.weight_magnitude_limit = 10000000
        self.mut_distribution = 3 #1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s

        #Task Params
        self.num_timesteps = 10
        self.time_delay = [0,0]
        self.num_food_items = 3
        self.num_drones = 1
        self.num_food_skus = 4
        self.num_poison_skus = [2,2] #Breaks down if all are poisonous

        #Dependents
        self.num_output = self.num_food_skus
        self.num_input = self.num_food_skus * 2
        if self.grumb_topology == 1: self.num_mem = self.num_hnodes
        self.save_foldername = 'R_Hive_mem/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)

        #Compute expected score for reasonable behavior
        self.expected_optimal = 0.0; self.expected_min = 0.0; self.expected_max = 0.0
        for num_poison in range(self.num_poison_skus[0], self.num_poison_skus[1]+1):
            ig_min = -1.0 * self.num_food_items * (num_poison)
            ig_max = 1.0 * self.num_food_items * (self.num_food_skus - num_poison)
            self.expected_min += ig_min; self.expected_max += ig_max
            score = self.num_food_items * (self.num_food_skus - num_poison) - num_poison
            self.expected_optimal += (score - ig_min) / (ig_max - ig_min)
        #Normalize by number of np.p choices
        self.expected_optimal /= ((self.num_poison_skus[1] + 1.0) - self.num_poison_skus[0])
        self.expected_max /= ((self.num_poison_skus[1] + 1.0) - self.num_poison_skus[0])
        self.expected_min /= ((self.num_poison_skus[1] + 1.0) - self.num_poison_skus[0])
        self.expected_optimal_translated = (self.expected_optimal * (self.expected_max-self.expected_min)+self.expected_min)

class Task_Forage:
    def __init__(self, parameters):
        self.parameters = parameters
        self.num_food_skus = parameters.num_food_skus; self.num_food_items = parameters.num_food_items; self.num_poison_skus = self.parameters.num_poison_skus
        self.num_drones = parameters.num_drones
        self.ssne = mod.Fast_SSNE(parameters)

        # Initialize food containers
        self.food_status = [self.num_food_items for _ in range(self.num_food_skus)] #Status of food (number left)
        self.food_poison_info = [False for _ in range(self.num_food_skus)]  # Status of whether food is poisonous

        #Initialize hives
        if self.parameters.load_colony: self.all_hives = self.load(self.parameters.save_foldername + 'colony')
        else:
            self.all_hives = []
            for hive in range(parameters.population_size): self.all_hives.append(mod.Hive(parameters))
            if self.parameters.load_colony: self.all_hives[0] = self.load(self.parameters.save_foldername + 'champion')

        self.hive_action = [[] for drone in range (self.num_drones)] #Track each drone's action set
        self.hive_local_reward = [[0.0 for sku_id in range (self.num_food_skus)] for drone in range (self.num_drones)]
        self.hive_delay = [0 for drone in range(self.num_drones)]  # Track if each drone is in time delay

    def reset_food_status(self):
        self.food_status = [self.num_food_items for _ in range(self.num_food_skus)]  # Status of food (number left)

    def reset_hive_delay(self):
        self.hive_delay = [0 for drone in range(self.num_drones)]  # Track if each drone is in time delay

    def reset_food_poison_info(self):
        for i in range(len(self.food_poison_info)):
            self.food_poison_info[i] = False #Reset everything to False

        #Randomly pick and assign food items as poisonous
        num_poisonous = randint(self.num_poison_skus[0], self.num_poison_skus[1])
        poison_ids = np.random.choice(self.num_food_skus, num_poisonous, replace=False)
        for item in poison_ids:
            self.food_poison_info[item] = True

        #Compute normalized score distribution
        min = -1.0 * self.parameters.num_food_items * (num_poisonous)
        max = 1.0 * self.parameters.num_food_items * (self.num_food_skus - num_poisonous)
        return min, max

    def reset_hive_local_reward(self):
        self.hive_local_reward = [[0.0 for sku_id in range(self.num_food_skus)] for drone in range(self.num_drones)]

    def take_action(self):
        self.reset_hive_local_reward() #Local reward to keep track of last observations
        temp_food_status = self.food_status[:]
        for drone_id in range(self.num_drones): #act with drones
            if self.hive_delay[drone_id] <= 0: #If not under time delay
                self.hive_delay[drone_id] = randint(self.parameters.time_delay[0], self.parameters.time_delay[1])
                action = self.hive_action[drone_id]
                if temp_food_status[action] != 0: #If anything left of the chosen food sku
                    self.food_status[action] -= 1 #Decrement food item of the sku chosen
                    if self.food_poison_info[action]: self.hive_local_reward[drone_id][action] -= 1.0
                    else: self.hive_local_reward[drone_id][action] += 1.0
            else: self.hive_delay[drone_id] -= 1

    def run_trial(self, hive):
        self.reset_food_status()
        hive.reset()
        self.reset_hive_delay()
        for timestep in range(self.parameters.num_timesteps):
            for drone_id in range(self.num_drones):
                if self.hive_delay[drone_id] > 0:
                    state = [0 for _ in range(self.num_food_skus)] + self.hive_local_reward[drone_id]
                    _ = hive.forward(state, drone_id)  # Run drones one step
                else:
                    state = self.food_status + self.hive_local_reward[drone_id]
                    action = hive.forward(state, drone_id) #Run drones one step
                    self.hive_action[drone_id] = action.index(max(action))
            self.take_action() #Move the entire hive up one step

        #Compute reward
        reward = 0.0
        for sku_id in range(self.num_food_skus):
            if self.food_status[sku_id] < 0: self.food_status[sku_id] = 0 #Bound the ones consumed to be zero
            if self.food_poison_info[sku_id]: #If food is poisonous
                reward -= 1.0 * (self.num_food_items - self.food_status[sku_id])
            else: reward += 1.0 * (self.num_food_items - self.food_status[sku_id])
        #print reward
        return reward

    def save(self, individual, filename ):
        mod.pickle_object(individual, filename)

    def load(self, filename):
        return mod.unpickle(filename)

    def evolve(self, gen, tracker):

        #Evaluation loop
        all_fitness = [[] for _ in range(self.parameters.population_size)]
        for eval_id in range(self.parameters.num_evals): #Multiple evals in different map inits to compute one fitness
            minimum, maximum = self.reset_food_poison_info()

            for hive_id, hive in enumerate(self.all_hives):
                fitness = self.run_trial(hive)
                all_fitness[hive_id].append((fitness-minimum)/(maximum-minimum))

        fitnesses = [sum(all_fitness[i])/self.parameters.num_evals for i in range(self.parameters.population_size)] #Average the finesses

        #Get champion index and compute validation score
        best_train_fitness = max(fitnesses)
        champion_index = fitnesses.index(best_train_fitness)

        #Run simulation of champion individual (validation_score)
        validation_fitness = 0.0
        for eval_id in range(self.parameters.num_evals):  # Multiple evals in different map inits to compute one fitness
            minimum, maximum = self.reset_food_poison_info()
            validation_fitness += (self.run_trial(self.all_hives[champion_index])-minimum) / ((maximum-minimum)*self.parameters.num_evals)

        #Save champion
        if gen % 100 == 0:
            ig_folder = self.parameters.save_foldername
            if not os.path.exists(ig_folder): os.makedirs(ig_folder)
            self.save(self.all_hives[champion_index], self.parameters.save_foldername + 'champion') #Save champion
            self.save(self.all_hives, self.parameters.save_foldername + 'colony')  # Save entire colony of hives (all population)
            self.save(tracker, self.parameters.save_foldername + 'tracker') #Save the tracker file
            np.savetxt(self.parameters.save_foldername + 'gen_tag', np.array([gen + 1]), fmt='%.3f', delimiter=',')


        #SSNE Epoch: Selection and Mutation/Crossover step
        self.ssne.epoch(self.all_hives, fitnesses)

        return best_train_fitness, validation_fitness

    def visualize(self):

        grid = [['-' for _ in range(self.dim_x)] for _ in range(self.dim_y)]

        #Draw in hive
        drone_symbol_bank = ["@",'#','$','%','&']
        for drone_pos, symbol in zip(self.hive_pos, drone_symbol_bank):
            x = int(drone_pos[0]); y = int(drone_pos[1])
            grid[x][y] = symbol


        symbol_bank = ['Q', 'W', 'E', 'R', 'T', 'Y']
        poison_symbol_bank = ['1', "2", '3', '4','5','6']
        #Draw in food
        for sku_id in range(self.num_foodskus):
            if self.food_poison_info[sku_id]: #If poisionous
                symbol = poison_symbol_bank.pop(0)
            else: symbol = symbol_bank.pop(0)
            for item_id in range(self.num_food_items):
                x = int(self.food_list[sku_id][item_id][0]); y = int(self.food_list[sku_id][item_id][1]);
                grid[x][y] = symbol

        for row in grid:
            print row
        print

if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    if parameters.load_colony:
        gen_start = int(np.loadtxt(parameters.save_foldername + 'gen_tag'))
        tracker = mod.unpickle(parameters.save_foldername + 'tracker')
    else:
        tracker = Tracker(parameters, ['best_train', 'valid', 'valid_translated'], '_hive_mem.csv')  # Initiate tracker
        gen_start = 1
    print 'Hive Memory Training with', parameters.num_input, 'inputs,', parameters.num_hnodes, 'hidden_nodes', parameters.num_output, 'outputs and', parameters.output_activation if parameters.output_activation == 'tanh' or parameters.output_activation == 'hardmax' else 'No output activation', 'Exp_opt:', '%.2f'%parameters.expected_optimal, 'Exp_min:', parameters.expected_min,'Exp_max:', parameters.expected_max
    sim_task = Task_Forage(parameters)
    for gen in range(gen_start, parameters.total_gens):
        best_train_fitness, validation_fitness = sim_task.evolve(gen, tracker)
        print 'Gen:', gen, 'Ep_best:', '%.2f' %best_train_fitness, ' Valid_Fit:', '%.2f' %validation_fitness, 'Cumul_valid:', '%.2f'%tracker.all_tracker[1][1], 'translated to', '%.2f'%tracker.all_tracker[2][1], 'out of', '%.2f'%parameters.expected_optimal_translated
        tracker.update([best_train_fitness, validation_fitness, (validation_fitness * (parameters.expected_max-parameters.expected_min)+parameters.expected_min)], gen)















