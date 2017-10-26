import numpy as np, os, math
import mod_hive_mem as mod, sys
from random import randint
#TODO Tracker loading seed backtrack to erase multiple y for time.

class Tracker(): #Tracker
    def __init__(self, parameters):
        self.foldername = parameters.save_foldername
        self.fitnesses = []; self.avg_fitness = 0; self.tr_avg_fit = []
        self.hof_fitnesses = []; self.hof_avg_fitness = 0; self.hof_tr_avg_fit = []
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)
        self.file_save = 'Hive_Mem.csv'

        if parameters.load_seed:
            self.tr_avg_fit = np.loadtxt(parameters.save_foldername + 'champ_train' + self.file_save, delimiter=',').tolist()
            self.hof_tr_avg_fit = np.loadtxt(parameters.save_foldername + 'champ_valid' + self.file_save, delimiter=',').tolist()


    def add_fitness(self, fitness, generation):
        self.fitnesses.append(fitness)
        if len(self.fitnesses) > 100:
            self.fitnesses.pop(0)
        self.avg_fitness = sum(self.fitnesses)/len(self.fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + 'champ_train' + self.file_save
            self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
            np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

    def add_hof_fitness(self, hof_fitness, generation):
        self.hof_fitnesses.append(hof_fitness)
        if len(self.hof_fitnesses) > 100:
            self.hof_fitnesses.pop(0)
        self.hof_avg_fitness = sum(self.hof_fitnesses)/len(self.hof_fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + 'champ_valid' + self.file_save
            self.hof_tr_avg_fit.append(np.array([generation, self.hof_avg_fitness]))
            np.savetxt(filename, np.array(self.hof_tr_avg_fit), fmt='%.3f', delimiter=',')

    def save_csv(self, generation, filename):
        self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
        np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

class Parameters:
    def __init__(self):
        self.population_size = 100
        self.load_seed = 0
        self.load_colony = 0
        self.total_gens = 50
        self.is_hive_mem = True #Is Hive memory connected/active? If not, no communication between the agents
        self.num_evals = 10 #Number of different maps to run each individual before getting a fitness

        #NN specifics
        self.num_hnodes = 25
        self.num_mem = 10
        self.grumb_topology = 3 #1: Default (hidden nodes cardinality attached to that of mem (No trascriber))
                                #2: Detached (Memory independent from hidden nodes (transcribing function))
                                #3: FF (Normal Feed-Forward Net)
        self.output_activation = 'tanh' #tanh or hardmax

        #SSNE stuff
        self.elite_fraction = 0.03
        self.crossover_prob = 0.05
        self.mutation_prob = 0.9
        self.homogenize_prob = 0.005
        self.homogenize_gates_prob = 0.05
        self.hive_crossover_prob = 0.03
        self.extinction_prob = 0.004 #Probability of extinction event
        self.extinction_magnituide = 0.5 #Probabilty of extinction for each genome, given an extinction event
        self.weight_magnitude_limit = 10000000
        self.mut_distribution = 3 #1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s

        #Task Params
        self.num_timesteps = 10
        self.num_food_items = 4
        self.num_drones = 1
        self.num_food_skus = 4
        self.num_poison_skus = 2

        #Dependents
        self.num_output = self.num_food_skus
        self.num_input = self.num_food_skus * 2
        if self.grumb_topology == 1: self.num_mem = self.num_hnodes
        self.save_foldername = 'R_Hive_mem/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)
        self.optimal_score = self.num_food_items * (self.num_food_skus - self.num_poison_skus) - (1.0 * self.num_poison_skus/self.num_food_skus) * self.num_poison_skus

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
            if self.parameters.load_seed: self.all_hives[0] = self.load(self.parameters.save_foldername + 'champion')


        self.hive_action = [[] for drone in range (self.num_drones)] #Track each drone's action set
        self.hive_local_reward = [[0.0 for sku_id in range (self.num_food_skus)] for drone in range (self.num_drones)]

    def reset_food_status(self):
        self.food_status = [self.num_food_items for _ in range(self.num_food_skus)]  # Status of food (number left)

    def reset_food_poison_info(self):
        for i in range(len(self.food_poison_info)):
            self.food_poison_info[i] = False #Reset everything to False

        #Randomly pick and assign food items as poisonous
        poison_ids = np.random.choice(self.num_food_skus, self.num_poison_skus, replace=False)
        for item in poison_ids:
            self.food_poison_info[item] = True

    def reset_hive_local_reward(self):
        self.hive_local_reward = [[0.0 for sku_id in range(self.num_food_skus)] for drone in range(self.num_drones)]

    def take_action(self):
        self.reset_hive_local_reward() #Local reward to keep track of last observations
        temp_food_status = self.food_status[:]
        for drone_id in range(self.num_drones): #act with drones
            action = self.hive_action[drone_id]
            if temp_food_status[action] != 0: #If anything left of the chosen food sku
                self.food_status[action] -= 1 #Decrement food item of the sku chosen
                if self.food_poison_info[action]: self.hive_local_reward[drone_id][action] -= 1.0
                else: self.hive_local_reward[drone_id][action] += 1.0

    def run_trial(self, hive):
        self.reset_food_status()
        hive.reset()
        for timestep in range(self.parameters.num_timesteps):
            for drone_id in range(self.num_drones):
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

        return reward

    def save(self, individual, filename ):
        mod.pickle_object(individual, filename)

    def load(self, filename):
        return mod.unpickle(filename)

    def evolve(self, gen):

        #Evaluation loop
        all_fitness = [[] for _ in range(self.parameters.population_size)]
        for eval_id in range(self.parameters.num_evals): #Multiple evals in different map inits to compute one fitness
            self.reset_food_poison_info()

            for hive_id, hive in enumerate(self.all_hives):
                fitness = self.run_trial(hive)
                all_fitness[hive_id].append(fitness)

        fitnesses = [sum(all_fitness[i])/self.parameters.num_evals for i in range(self.parameters.population_size)] #Average the finesses

        #Get champion index and compute validation score
        best_train_fitness = max(fitnesses)
        champion_index = fitnesses.index(best_train_fitness)

        #Run simulation of champion individual (validation_score)
        validation_fitness = 0.0
        for eval_id in range(self.parameters.num_evals):  # Multiple evals in different map inits to compute one fitness
            self.reset_food_poison_info()
            validation_fitness += self.run_trial(self.all_hives[champion_index])/(self.parameters.num_evals)

        #Save champion
        if gen % 100 == 0:
            ig_folder = self.parameters.save_foldername
            if not os.path.exists(ig_folder): os.makedirs(ig_folder)
            self.save(self.all_hives[champion_index], self.parameters.save_foldername + 'champion') #Save champion
            self.save(self.all_hives, self.parameters.save_foldername + 'colony')  # Save entire colony of hives (all population)
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
    tracker = Tracker(parameters)  # Initiate tracker
    print 'Hive Memory Training with', parameters.num_input, 'inputs,', parameters.num_hnodes, 'hidden_nodes', parameters.num_output, 'outputs and', parameters.output_activation if parameters.output_activation == 'tanh' or parameters.output_activation == 'hardmax' else 'No output activation'

    sim_task = Task_Forage(parameters)
    if parameters.load_seed: gen_start = int(np.loadtxt(parameters.save_foldername + 'gen_tag'))
    else: gen_start = 1
    for gen in range(gen_start, parameters.total_gens):
        best_train_fitness, validation_fitness = sim_task.evolve(gen)
        print 'Gen:', gen, 'Ep_best:', '%.2f' %best_train_fitness, ' Valid_Fit:', '%.2f' %validation_fitness, 'Cumul_valid:', '%.2f'%tracker.hof_avg_fitness, ' out of', '%.2f'%parameters.optimal_score
        tracker.add_fitness(best_train_fitness, gen)  # Add best global performance to tracker
        tracker.add_hof_fitness(validation_fitness, gen)  # Add validation global performance to tracker














