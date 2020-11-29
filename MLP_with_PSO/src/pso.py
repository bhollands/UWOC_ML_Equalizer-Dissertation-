import numpy as np
import pandas as pd

'''
particel class
'''
class Particle:
    def __init__(self, dimensions, position_range,velocity_range):
        self.position = np.random.uniform(position_range[0], position_range[1], (dimensions,))#create random positions
        self.velocity = np.random.uniform(velocity_range[0], velocity_range[1], (dimensions, ))#create random velocity
        self.pbFit = np.inf #personal best fitness is infinity
        self.pbPos = np.zeros((dimensions, )) #personal best position

'''
swarm class
'''
class Swarm:
    def __init__(self, num_of_particles, dimensions, position_range, 
                velocity_range, weight_range, c):
    
        self.particles      = np.array([Particle(dimensions, position_range,velocity_range) for i in range(num_of_particles)]) #creates an array with all the particle
        self.globalBestFit  = np.inf #set global best position to infinity
        self.globalBestPos  = np.zeros((dimensions))
        self.position_range = position_range #set the max range of position values
        self.velocity_range = velocity_range # set the max range of velocity values
        self.weight_range   = weight_range # set the weight range
        self.cog            = c[0] #set cognitive
        self.social         = c[1] #set social factor
        self.dimensions     = dimensions # set the number of dimensions
        
    '''
    searches for the best partivle position
    '''
    def search(self, feedForward, X,y, print_step, epochs):
        fitness_array = np.zeros(epochs)
        for i in range(epochs): #loop for the number of epochs
            for cur_particle in self.particles: #for each particle in the swarm
                fitness = feedForward(X, y, cur_particle.position) #get the fitness for the current position(needs to be run through the NN)

                #want fitness to be as small as possible
                if fitness < cur_particle.pbFit: #if the fitness is smaller than the current best
                    cur_particle.pbFit = fitness #the personal bets
                    cur_particle.pbPos = cur_particle.position.copy()

                if fitness < self.globalBestFit: #if fitness less than the glohal best
                    self.globalBestFit = fitness #updat eglobal fitness
                    self.globalBestPos = cur_particle.position.copy() #save the position

            for cur_particle in self.particles: #for each particl in the swarm update position and velocity
                #randomise the inertia weight
                inertia_weight = np.random.uniform(self.weight_range[0], self.weight_range[1], 1)[0]
                #print("inertia weight", inertia_weight)
                cog = self.cog  #cognition factor
                social = self.social #social factor
                dim = self.dimensions #number of dimensions
                r1 = np.random.uniform(0.0, 1.0, (dim, )) #random value
                r2 = np.random.uniform(0.0, 1.0, (dim, )) #random value
                pBestPos = cur_particle.pbPos #partivles bets position
                currPos = cur_particle.position #particles current position
                currVel = cur_particle.velocity #particels current velocity
                globalBestPos = self.globalBestPos #global best position

                #update velocity eqation
                cur_particle.velocity = (inertia_weight*currVel)*(cog*r1*(pBestPos-currPos)) + (social*r2*(globalBestPos - currPos))

                #update position equation
                cur_particle.position = cur_particle.position +cur_particle.velocity

            if i % print_step == 0: #print the fitness
                print('Epoch#: ', i+1, fitness)
            
            fitness_array[i] = fitness
        
        return fitness_array

    '''
    returns best particle position
    '''
    def get_best_solution(self):
        return self.globalBestPos