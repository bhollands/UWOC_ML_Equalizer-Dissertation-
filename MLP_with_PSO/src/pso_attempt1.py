import numpy as np
import random 

swarmsize = 10
personal_best = 0
informants_best = 0
global_best = 0
jump_size = 1

class Particle():
    def __init__(self):
        self.position=[]          # particle position
        self.velocity=[]          # particle velocity
        self.global_best = 0

        for i in range(0,swarmsize):
            self.velocity.append([random.uniform(-1,1)])
            self.position.append([float(i)])
            # print(self.velocity)
            # print(self.position)

    # def fitness(self, X, y):
    #     for i in range(0, swarmsize):
    #         self.mean_squared_error = np.square(np.subtract(X[i], y[i])).mean()

    #         if self.global_best == 0:
    #             self.global_best = self.mean_squared_error

    #         elif self.mean_squared_error > self.global_best:
    #             self.global_best = self.mean_squared_error

    #         elif self.mean_squared_error < self.global_best:
    #             self.global_best = self.global_best

    #     print(self.global_best)


    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0, swarmsize):
            self.position = [[x + y for x,y in zip(l1,l2)] for l1,l2 in zip(self.position, self.velocity)]  # add position and velocity to get new particle position

            # print(self.position[i][0])

            # adjust maximum position if necessary
            if self.position[i][0] > bounds[0][1]:
                self.position[i][0] = bounds[0][1]

            # adjust minimum position if necessary
            elif self.position[i][0] < bounds[0][0]:
                self.position[i][0] = bounds[0][0]

            # do not adjust position if within range
            elif bounds[0][0] < self.position[i][0] < bounds[0][1]:
                self.position[i][0] = self.position[i][0]

      
class PSO():    # not finished implementing and gives errors
    def __init__(self,epochs,X,y,bounds):

        # establish the swarm
        swarm=[]
        for i in range(0,swarmsize):
            swarm.append(Particle().position)   # append particle position to swarm

        # begin optimization loop
        for i in range(0,epochs):
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                fitness = swarm[j].feedForward(X, y, Particle().position) #get the fitness for current position - run through ANN

                #want fitness to be as small as possible
                if fitness < current_particle.personal_best: #if the fitness is less than the current personal best
                    current_particle.personal_best = fitness #the personal best is the current fitness
                    current_particle.personal_best = current_particle.position.copy() #copy personal best to update particles position

                if fitness < self.global_best: #if the fitness is less than the global best
                    self.global_best = fitness #update global best to current fitness
                    self.global_best = current_particle.position.copy() #copy global best to update particles position

            

def main():

    X = [1,2,1,5,3,2,2,1,3,6]                           #prediction
    y = [0.3,0.4,0.6,0.1,0.3,0.4,0.6,0.1,0.7,0.3]       #actual answers
    bounds = [(-5.0,5.0)]
    epochs = 50
    pso = Particle()
    # print(pso.fitness(X,y))
    # print(pso.update_position(bounds))

if __name__ == "__main__":
    main()
