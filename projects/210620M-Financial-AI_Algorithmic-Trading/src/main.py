import pandas as pd
from model import train_model


train = pd.read_csv('processed_data/train.csv')
test = pd.read_csv('processed_data/test.csv')


model = train_model(algo='A2C', timesteps=100000)