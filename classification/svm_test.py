from sklearn import svm
from sklearn.metrics import confusion_matrix
from scipy import io
from sklearn.model_selection import train_test_split

# X = [[0], [1], [2], [3]]
# Y = [0, 1, 2, 3]
mat_path = r'D:\Projects\IoT_recognition\20181111\vis_2k\global_color_rec_01.mat'
digits = io.loadmat(mat_path)
X, Y = digits.get('feature_matrix'), digits.get('label')[0]  # X: nxm: n=500//sample, m=12,10,71,400//feature
#X=X[::5]
#Y=Y[::5]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
clf.fit(X_train, Y_train)

svm_prediction = clf.predict(X_test)
accuracy = clf.score(X_test, Y_test)
print(accuracy)
cm = confusion_matrix(Y_test, svm_prediction)
print(cm)
