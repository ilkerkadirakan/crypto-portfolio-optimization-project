from scipy.io import loadmat
import pandas as pd
import numpy as np

loaded_data = loadmat("data/raw/AAVEBTC1.mat")

print(loaded_data.keys())

print(loaded_data.values())
df = pd.DataFrame(columns=["price"], data=loaded_data.get('price'))
print(df.head())