import matplotlib.pyplot as plt
from scipy import io
import numpy as np
from scipy import io as sio

mat_path = r'D:\Projects\IoT_recognition\20181111\vis_2k\global_color.mat'
x_axis_index = 0
y_axis_index = 1
digits = io.loadmat(mat_path)
X, y = digits.get('feature_matrix'), digits.get('label')[0]  # X: nxm: n=500//sample, m=12,10,71,400//feature
# X=X.T[1:].T
n_samples, n_features = X.shape

# eigenvalues, eigenvectors = np.linalg.eig(np.cov(X))  # values: nx1/67x1, vectors: nxn/67x67

U, s, Vh = np.linalg.svd(X.transpose(), full_matrices=False)  # u: mxm, s: mx1, v:nxn/67x67
s[:1] = 0

reconstructed_x = U.dot(np.diag(s)).dot(Vh)

fig = plt.figure()
fig.subplots_adjust(bottom=0.05)
fig.subplots_adjust(top=0.95)
fig.subplots_adjust(hspace=0.35)

ax = fig.add_subplot(321)
ax.imshow(X.transpose())
ax.set_aspect(30)
if "raw" in mat_path:
    ax.set_aspect(0.01)
ax.set_title('original_mat')

ax = fig.add_subplot(322)
ax.bar(np.arange(len(s)), s)
ax.set_title('singular_values_feature')

ax = fig.add_subplot(323)
ax.imshow(reconstructed_x)
ax.set_aspect(30)
if "raw" in mat_path:
    ax.set_aspect(0.01)
ax.set_title('original_mat')
plt.show()

save_path = mat_path.split('.')[0] + "_rec_01.mat"
sio.savemat(save_path, mdict={'feature_matrix': reconstructed_x.T, 'label': y})
