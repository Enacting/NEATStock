import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



import NEAT_simulate
import NN
import yfinance as yf
tickerSymbol = 'MSFT'
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(period='1d', start='2010-1-1', end='2020-1-25')
x = list(tickerDf["Close"])

close = x
initial_money = 10000
window_size = 30
skip = 1

population_size = 100
generations = 100
mutation_rate = 0.1
neural_evolve = NEAT_simulate.neuro_evolution(population_size, mutation_rate, NN.neuralnetwork,
                              window_size, window_size, close, skip, initial_money)
fittest_nets = neural_evolve.evolve(50)
states_buy, states_sell, total_gains, invest, action = neural_evolve.apply_action(fittest_nets)

#Risk shows the spontaneity of dataset because it shows the difficulty the system had in predicting it. 
#The accuracy varies and if the risk is above 50% then you're better off doing the opposite of what it says. 
#The closer it is to 50%. the worse it is generally at predicting. 

if action == 1:
  act = "Buy"
else:
  act = "Sell"
  
print("I recommend you {}".format(act))
print("The risk of investing in this stock is: {}".format(str(100-invest)))
