import sys
import pandas as pd
import A2
import matplotlib.pyplot as plt
import warnings
import os

warnings.simplefilter(action='ignore')

# Plot settings
plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams["axes.grid"] = True
plt.rcParams['grid.linestyle'] = '-.'
plt.rcParams['grid.linewidth'] = 0.4


##  Solves the model with 'load_samples.csv' and 'wind_samples.csv'
solve = 1
if solve:
    uc = A2.unit_comitment()

    uc.T = len(uc.load_samples)
    uc.wf_power = uc.wind_samples
    uc.demand = uc.load_samples

    result = uc.solve_model()
    result = pd.concat([result, uc.wind_samples, uc.load_samples ], axis=1)
    print(result.to_string())
    result.to_csv(os.path.join('input', 'processed', 'model_data.csv'), index=False, sep=';')


## Some plots for analysis
data = pd.read_csv(os.path.join('input', 'processed', 'model_data.csv'), sep=';')
print(data.columns)

data = data.loc[:200]

data.total_load.plot()
data.gen3.plot()


plt.title('Gen3 Power')
#data.u2.plot()
#data.u3.plot()
plt.show()



