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

        self.population_size = 50
        self.load_seed = False
        self.total_gens = 10000
        self.is_hive_mem = True #Is Hive memory connected/active? If not, no communication between the agents
        self.num_evals = 10 #Number of different maps to run each individual before getting a fitness

        #NN specifics

        self.num_hnodes = 40
        self.memory_size = self.num_hnodes


        #SSNE stuff
        self.elite_fraction = 0.04
        self.crossover_prob = 0.05
        self.mutation_prob = 0.75
        self.extinction_prob = 0.004 #Probability of extinction event
        self.extinction_magnituide = 0.5 #Probabilty of extinction for each genome, given an extinction event
        self.weight_magnitude_limit = 10000000
        self.mut_distribution = 3 #1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s

        #Task Params
        self.dim_x = 10; self.dim_y = 10; self.angle_res = 45; self.obs_dist = 1.0
        self.is_random_spawn = True
        self.num_timesteps = 20
        self.num_food_skus = 3
        self.num_food_items = 3
        self.num_drones = 1
        self.num_poison_skus = 1

        #Dependents
        self.num_input = (360 / self.angle_res) * (self.num_food_skus * 3 + 2)
        self.num_output = 2
        self.save_foldername = 'R_Hive_mem/'
        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)

class Task_Forage:
    def __init__(self, parameters):
        self.parameters = parameters
        self.dim_x = parameters.dim_x; self.dim_y = parameters.dim_y; self.is_random_spawn = parameters.is_random_spawn
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

        if self.is_random_spawn:
            for sku_id, sku in enumerate(self.food_list):
                for i, item in enumerate(sku):
                    if i % self.parameters.num_food_items == 0:
                        x = randint(start, center - rad - 1)
                        y = randint(start, end)
                    elif i % self.parameters.num_food_items == 1:
                        x = randint(center + rad + 1, end)
                        y = randint(start, end)
                    elif i % self.parameters.num_food_items == 2:
                        x = randint(center - rad, center + rad)
                        y = randint(start, center - rad - 1)
                    else:
                        x = randint(center - rad, center + rad)
                        y = randint(center + rad + 1, end)
                    self.food_list[sku_id][i] = (x,y)

        #TODO
        #else:  # Pre-defined starting positions

    def reset_food_status(self):
        self.food_status = [[False for item in range(self.parameters.num_food_items)] for sku in
                            range(self.parameters.num_food_skus)]

    def reset_food_poison_info(self):
        for item in self.food_poison_info: item = False #Reset everything to False

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

    def get_state(self, drone_id):  # Returns a flattened array around the predator position
        self_x = self.hive_pos[drone_id][0]; self_y = self.hive_pos[drone_id][1]

        #Angle brackets representation
        state = np.zeros(((360 / self.angle_res), self.num_foodskus * 3 + 2)) #FORMAT: [bracket] = (drone_avg_dist, drone_number,
                                                                                                    #food_avg_dist, food_number_item, reward ......]
        temp_food_dist_list = []
        for sku_id in range(self.num_foodskus):
            temp_food_dist_list.append([[] for _ in xrange(360 / self.angle_res)])
        temp_drone_dist_list = [[] for _ in xrange(360 / self.angle_res)]

        #Log all distance into brackets for food
        for sku_id in range(self.num_foodskus):
            for item_id in range(self.num_food_items):
                if self.food_status[sku_id][item_id] == False: #Only if not accessed/observed yet
                    x1 = self.food_list[sku_id][item_id][0] - self_x; x2 = -1.0
                    y1 = self.food_list[sku_id][item_id][1] - self_y; y2 = 0.0
                    angle, dist = self.get_angle_dist(x1, y1, x2, y2)
                    bracket = int(angle / self.angle_res)
                    temp_food_dist_list[sku_id][bracket].append(dist)
                    if dist <= self.obs_dist: #Reward info
                        iter_pos = 2 + sku_id * 3
                        state[bracket][iter_pos + 2] += self.food_poison_info[sku_id]
                        self.food_status[sku_id][item_id] = True

        # Log all distance into brackets for other drones
        for other_drone_id in range(self.num_drones):
            if other_drone_id != drone_id: #Not the drone itself (don't count itself)
                x1 = self.hive_pos[other_drone_id][0] - self_x; x2 = -1.0
                y1 = self.hive_pos[other_drone_id][1] - self_y; y2 = 0.0
                angle, dist = self.get_angle_dist(x1, y1, x2, y2)
                bracket = int(angle / self.angle_res)
                temp_drone_dist_list[bracket].append(dist)

        ####Encode the information onto the state
        for bracket in range(int(360 / self.angle_res)):
            #Drones
            state[bracket][1] = len(temp_drone_dist_list[bracket])
            if state[bracket][1] > 0:
                state[bracket][0] = sum(temp_drone_dist_list[bracket])/len(temp_drone_dist_list[bracket])

            #Foods
            for sku_id in range(self.num_foodskus):
                iter_pos = 2 + sku_id * 3
                state[bracket][iter_pos + 1] = len(temp_food_dist_list[sku_id][bracket])
                if state[bracket][iter_pos + 1] > 0:
                    state[bracket][iter_pos] = sum(temp_food_dist_list[sku_id][bracket])/len(temp_food_dist_list[sku_id][bracket])


        state = state.flatten().tolist()
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
        for timestep in range(self.parameters.num_timesteps):
            for drone_id in range(self.num_drones):
                state = self.get_state(drone_id)
                action = hive.forward(state, drone_id) #Run drones one step
                self.hive_action[drone_id][0], self.hive_action[drone_id][1] = action[0], action[1]
            self.move() #Move the entire hive up one step

        #Compute reward
        reward = 0.0
        for sku_id in range(self.num_foodskus):
            for item_id in range(self.num_food_items):
                if self.food_status[sku_id][item_id]: #If food is accessed
                    if self.food_poison_info[sku_id]: #If food is poisonous
                        reward += -1.0
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













