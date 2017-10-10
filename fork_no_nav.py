import numpy as np, os, math
import mod_hive_mem as mod, sys
from random import randint


class Tracker(): #Tracker
    def __init__(self, parameters):
        self.foldername = parameters.save_foldername
        self.fitnesses = []; self.avg_fitness = 0; self.tr_avg_fit = []
        self.hof_fitnesses = []; self.hof_avg_fitness = 0; self.hof_tr_avg_fit = []
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)
        self.file_save = 'Hive_Mem.csv'

    def add_fitness(self, fitness, generation):
        self.fitnesses.append(fitness)
        if len(self.fitnesses) > 50:
            self.fitnesses.pop(0)
        self.avg_fitness = sum(self.fitnesses)/len(self.fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + 'champ_train' + self.file_save
            self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
            np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

    def add_hof_fitness(self, hof_fitness, generation):
        self.hof_fitnesses.append(hof_fitness)
        if len(self.hof_fitnesses) > 50:
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
        self.load_seed = False
        self.total_gens = 100000
        self.is_hive_mem = True #Is Hive memory connected/active? If not, no communication between the agents
        self.num_evals = 10 #Number of different maps to run each individual before getting a fitness

        #NN specifics
        self.num_hnodes = 30
        self.memory_size = self.num_hnodes


        #SSNE stuff
        self.elite_fraction = 0.04
        self.crossover_prob = 0.05
        self.mutation_prob = 0.9
        self.extinction_prob = 0.004 #Probability of extinction event
        self.extinction_magnituide = 0.5 #Probabilty of extinction for each genome, given an extinction event
        self.weight_magnitude_limit = 10000000
        self.mut_distribution = 3 #1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s

        #Task Params
        self.dim_x = 10; self.dim_y = 10; self.obs_dist = 1
        self.num_timesteps = 15
        self.num_food_items = 10
        self.num_drones = 10
        self.num_food_skus = 10
        self.num_poison_skus = 5

        #State representation
        self.angle_res = 45;
        self.state_representation = 3 #1: Bracketed with [avg dist, cardinality, reward]
                                      #2: Bracketed with [avg dist, min_dist, cardinality, reward]
                                      #3: All drones and food listed (full observability) [dist, angle, reward]

        self.food_spawn_protocol = 2 #1: Localized SKUs
                                     #2: Full Random

        #Dependents
        if self.state_representation == 1: self.num_input = (360 / self.angle_res) * (self.num_food_skus * 3 + 2)
        if self.state_representation == 2: self.num_input = (360 / self.angle_res) * (self.num_food_skus * 4 + 3)
        if self.state_representation == 3: self.num_input = (self.num_food_skus * self.num_food_items* 2)
        self.num_output = self.num_food_skus * self.num_food_items
        self.save_foldername = 'R_Hive_mem/'
        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)

class Task_Forage:
    def __init__(self, parameters):
        self.parameters = parameters
        self.dim_x = parameters.dim_x; self.dim_y = parameters.dim_y
        self.num_foodskus = parameters.num_food_skus; self.num_food_items = parameters.num_food_items; self.num_poison_skus = self.parameters.num_poison_skus
        self.num_drones = parameters.num_drones; self.angle_res = parameters.angle_res #Angle resolution
        self.obs_dist = parameters.obs_dist #Observation radius requirements
        self.ssne = mod.Fast_SSNE(parameters)

        # Initialize food containers
        self.food_list = [[[0.0,0.0] for item in range (self.parameters.num_food_items)] for sku in range(self.parameters.num_food_skus)] #FORMAT: [sku][item] = (x, y) tuple
        self.food_status = [[False for item in range (self.parameters.num_food_items)] for sku in range(self.parameters.num_food_skus)] #Status of is food accessed
        self.food_poison_info = [False for _ in range(self.num_foodskus)]  # Status of whether food is poisonous


        #Initialize hives
        self.all_hives = []
        for hive in range(parameters.population_size):
            self.all_hives.append(mod.Hive(parameters))
        if self.parameters.load_seed: self.all_hives[0] = self.load(self.parameters.save_foldername + 'champion')
        self.hive_pos = [[0.0,0.0] for drone in range (self.num_drones)] #Track each drone's position
        self.hive_action = [[0.0, 0.0] for drone in range (self.num_drones)] #Track each drone's action set

    def reset_food_pos(self):
        start = 1.0;
        end = self.dim_x - 1.0
        rad = int(self.dim_x / math.sqrt(3) / 2.0)
        center = int((start + end) / 2.0)

        dist_ctrl = randint(0,3) #Distribution control temp variable
        for sku_id in range(self.parameters.num_food_skus):
            dist_ctrl += 1
            for i in range(self.parameters.num_food_items):
                if self.parameters.food_spawn_protocol == 2: dist_ctrl += randint(0, 3) #Random distribution
                if  dist_ctrl % 4 == 0:
                    x = randint(start, center - rad)
                    y = randint(start, end)
                elif dist_ctrl % 4 == 1:
                    x = randint(center + rad, end)
                    y = randint(start, end)
                elif dist_ctrl % 4 == 2:
                    x = randint(center - rad, center + rad)
                    y = randint(start, center - rad)
                else:
                    x = randint(center - rad, center + rad)
                    y = randint(center + rad, end)
                self.food_list[sku_id][i] = (x,y)

    def reset_food_status(self):
        self.food_status = [[False for item in range(self.parameters.num_food_items)] for sku in
                            range(self.parameters.num_food_skus)]

    def reset_food_poison_info(self):
        for i in range(len(self.food_poison_info)):
            self.food_poison_info[i] = False #Reset everything to False

        #Randomly pick and assign food items as poisonous
        poison_ids = np.random.choice(self.num_foodskus, self.num_poison_skus, replace=False)
        for item in poison_ids:
            self.food_poison_info[item] = True

    def reset_hive_pos(self):

        start = 1.0;
        end = self.dim_x - 1.0
        rad = int(self.dim_x / math.sqrt(3) / 2.0)
        center = int((start + end) / 2.0)

        for drone_id in range(self.num_drones):
            quadrant = drone_id % 4
            if quadrant == 0:
                x = center - 1 - (drone_id / 4) % (center - rad)
                y = center - (drone_id / (4*center - rad)) % (center - rad)
            if quadrant == 1:
                x = center + (drone_id / (4*center - rad)) % (center - rad)
                y = center - 1 + (drone_id / 4) % (center - rad)
            if quadrant == 2:
                x = center + 1 + (drone_id / 4) % (center - rad)
                y = center + (drone_id / (4*center - rad)) % (center - rad)
            if quadrant == 3:
                x = center - (drone_id / (4*center- rad)) % (center - rad)
                y = center+ 1 - (drone_id / 4) % (center- rad)
            self.hive_pos[drone_id] = [x,y]

    def get_state(self, drone_id, action):  # Returns a flattened array around the predator position
        state = [0.0] * self.parameters.num_input

        # Log all distance into brackets for food
        iter_pos = -2
        for sku_id in range(self.num_foodskus):
            for item_id in range(self.num_food_items):
                iter_pos += 2
                state[iter_pos] = self.food_status[sku_id][item_id]
                if action == iter_pos/2:  # Reward info
                    if self.food_poison_info[sku_id]: state[iter_pos + 1] -= 1.0
                    else: state[iter_pos + 1] += 1.0
                    self.food_status[sku_id][item_id] = True

        return state

    def get_angle_dist(self, x1, y1, x2, y2):  # Computes angles and distance between two predators relative to (1,0) vector (x-axis)
        dot = x2 * x1 + y2 * y1  #dot product
        det = x2 * y1 - y2 * x1  # determinant
        angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
        angle = math.degrees(angle) + 180.0 + 270.0
        angle = angle % 360
        dist = x1 * x1 + y1 * y1
        dist = math.sqrt(dist)
        return angle, dist

    def get_dist(self, pos_1, pos_2):
        #Remmeber unlike the dist calculated in get_ang_dist function, this one computes directly from position not vectors
        return math.sqrt((pos_1[0]-pos_2[0])* (pos_1[0]-pos_2[0]) + (pos_1[1]-pos_2[1])* (pos_1[1]-pos_2[1]))

    def move(self):
        for drone_id in range(self.num_drones): #Move drones
            next_pos = [self.hive_pos[drone_id][0] + self.hive_action[drone_id][0], self.hive_pos[drone_id][1] + self.hive_action[drone_id][1]] #Compute next candidate position

            # Implement bounds
            if next_pos[0] >= self.dim_x-1: next_pos[0] = self.dim_x - 2
            elif next_pos[0] < 1: next_pos[0] = 1
            if next_pos[1] >= self.dim_y-1: next_pos[1] = self.dim_y - 2
            elif next_pos[1] < 1: next_pos[1] = 1

            #Update
            self.hive_pos[drone_id][0] = next_pos[0]; self.hive_pos[drone_id][1] = next_pos[1]

    def soft_reset(self):
        self.reset_food_status()
        self.reset_hive_pos()

    def hard_reset(self):
        self.soft_reset()
        self.reset_food_pos()
        self.reset_food_poison_info()

    def run_trial(self, hive):
        self.soft_reset()
        hive.reset()
        action = None
        for timestep in range(self.parameters.num_timesteps):
            #self.visualize()
            #raw_input('Continue')
            for drone_id in range(self.num_drones):
                state = self.get_state(drone_id, action)
                action = hive.forward(state, drone_id) #Run drones one step
                action = action.index(max(action))

        #Compute reward
        reward = 0.0
        for sku_id in range(self.num_foodskus):
            for item_id in range(self.num_food_items):
                if self.food_status[sku_id][item_id]: #If food is accessed
                    if self.food_poison_info[sku_id]: #If food is poisonous
                        reward -= 1.0
                    else: reward += 1.0

        return reward

    def save(self, individual, filename ):
        mod.pickle_object(individual, filename)

    def load(self, filename):
        return mod.unpickle(filename)

    def evolve(self, gen):

        #Evaluation loop
        all_fitness = [[] for _ in range(self.parameters.population_size)]
        for eval_id in range(self.parameters.num_evals): #Multiple evals in different map inits to compute one fitness
            self.hard_reset()

            for hive_id, hive in enumerate(self.all_hives):
                fitness = self.run_trial(hive)
                all_fitness[hive_id].append(fitness)

        fitnesses = [sum(all_fitness[i])/self.parameters.num_evals for i in range(self.parameters.population_size)] #Average the finesses

        #Get champion index and compute validation score
        best_train_fitness = max(fitnesses)
        champion_index = fitnesses.index(best_train_fitness)

        #Run simulation of champion individual (validation_score)
        validation_fitness = 0.0
        for eval_id in range(self.parameters.num_evals * 2):  # Multiple evals in different map inits to compute one fitness
            self.hard_reset()
            validation_fitness += self.run_trial(self.all_hives[champion_index])/(self.parameters.num_evals*2.0)

        #Save champion
        if gen % 100 == 0:
            ig_folder = self.parameters.save_foldername
            if not os.path.exists(ig_folder): os.makedirs(ig_folder)
            self.save(self.all_hives[champion_index], self.parameters.save_foldername + 'champion') #Save champion
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
    print 'Hive Memory Training '

    sim_task = Task_Forage(parameters)
    for gen in range(1, parameters.total_gens):
        best_train_fitness, validation_fitness = sim_task.evolve(gen)
        print 'Gen:', gen, 'Epoch_best:', best_train_fitness, ' Valid_Fitness:', validation_fitness
        tracker.add_fitness(best_train_fitness, gen)  # Add best global performance to tracker
        tracker.add_hof_fitness(validation_fitness, gen)  # Add validation global performance to tracker














