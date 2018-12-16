import matplotlib.pyplot as plt
from scipy import io
import numpy as np

mat_path = r'D:\Projects\IoT_recognition\20181028\vis\ratio_hue.mat'
digits = io.loadmat(mat_path)
X, y = digits.get('feature_matrix'), digits.get('label')[0]
n_samples, n_features = X.shape

ev1 = X.transpose()[0]
ev2 = X.transpose()[1]

fig = plt.figure()
for i in range(ev1.shape[0]):
    fig.text(ev1[i], ev2[i], str(y[i] + 1), color=plt.cm.tab20(int(y[i])), fontdict={'size': 8})
plt.show()