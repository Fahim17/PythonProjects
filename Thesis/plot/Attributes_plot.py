import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

X = np.loadtxt("X_main.dat")
y = np.loadtxt("y_main.dat")

# X = preprocessing.normalize(X, norm='l2')
sc = StandardScaler()
X = sc.fit_transform(X)
X = preprocessing.normalize(X, norm='l2')
print(X.shape)
y_temp_DI = np.array(X[:,221])
#y_temp_DII = np.array(X[:,235])
#y_temp_DIII = np.array(X[:,245])
x_temp = np.array(X[:,225])

#fig = plt.figure(figsize=(10, 8))
#ax = fig.add_subplot(111, projection='3d')
le = ['Normal', 'Disease']
plt.figure(figsize=(10, 8))
plt.title('Attribute Presentation[Scaled]')
plt.xlabel('P curve Amplitude')
plt.ylabel('R curve Amplitude')
c=0

for i in y:
	if(i==1):
		 plt.plot(x_temp[c],y_temp_DI[c] , 'g+')
		# plt.plot(x_temp[c],y_temp_DII[c] , 'g*')
#		ax.scatter(y_temp_DI, y_temp_DII, y_temp_DIII, c='g', marker='+')
	# elif(i==10):
	else:
		 plt.plot(x_temp[c],y_temp_DI[c] , 'r+')
		# plt.plot(x_temp[c],y_temp[c] , 'r*')
#		ax.scatter(y_temp_DI, y_temp_DII, y_temp_DIII, c='r', marker='^')
	c+=1
plt.plot(x_temp[0],y_temp_DI[0] , 'g+',label=le[0])
plt.plot(x_temp[0],y_temp_DI[0] , 'r+', label=le[1])
plt.legend(loc = 4)
# plt.plot(x_temp,y_temp , 'go')
# plt.axis([20, 100, 1, 12.5])
plt.show()
