"""
Create dataset.npz from x and y train .csv files. 
"""
from src.utils.csv2npz import csv2npz

csv2npz('datasets/x_train_LsAZgHU.csv', 'datasets/y_train_EFo1WyE.csv', 'datasets', \
 'dataset_CAPT_v7')
