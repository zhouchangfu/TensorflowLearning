from scipy import stats
import numpy as np
import time
a = np.array([0, 0, 0.0, 1, 1, 1, 1])
b = np.arange(7)
result=stats.pearsonr(a, b)
a,b=result
print(result)
print(time.mktime(time.strptime("2019-02-26",'%Y-%m-%d')))