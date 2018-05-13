

from scipy.stats import probplot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = '/home/leonard/Desktop/Workspace/Kaggle/HousingPrices'
train_data = pd.read_csv('{}/data/train.csv'.format(ROOT_DIR), index_col=0)
target = train_data['SalePrice']
target = np.log(target)

quantile_data, fit_params = probplot(target)
slope, intercept, r = fit_params

plt.scatter(*quantile_data)
x = np.linspace(-3, 3, 1000)
plt.plot(x, intercept + slope * x, color='red')
plt.show()
