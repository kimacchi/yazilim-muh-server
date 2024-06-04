import pandas as pd 
import numpy as np 
from scipy import stats


dataset = pd.read_csv(r"./dataset/House_Rent_Dataset.csv")
dataset=dataset[np.abs(stats.zscore(dataset["Rent"])) < 3]


for column in dataset.columns:
    print(f"'{column}': {dataset[column].unique()},\n")