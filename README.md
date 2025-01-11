This repository contains various seat price optimisation models, and a simulation model to test them against. 

All seat price optimisation models have their own class and file, and are called in the same manner (files like LinPred.py, QLearning,py, ExpPred.py, etc.). The simulation code is in SeatSimulation.py, where customer characteristics are sampled. The main.py file calls all other models and the simulation. Also in there the parameters for the simulation are controlled. 
