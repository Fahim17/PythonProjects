import numpy as np
from io import StringIO

instance = 452-10

############################## reading data from csv file ##############################
f = open("arrhythmia.data")
data = np.ones([280])

for i in range(instance):
    s = f.readline()    
    line = np.genfromtxt(StringIO(s), delimiter=",")
    data = np.vstack((data,line))
data = data[1:]
# print(np.sum(data))

############################## ########################## ##############################

############################## Spliting X and Y ##############################
X = np.delete(data,[279],axis=1)
y = data[:,279]
# print(X)
# print(y)

############################## ########################## ##############################

############################## Cleaning up the data ##############################
# X_mod = np.copy(X)
# X_mod = np.delete(X,[4,66,91,200,213,238,242,360,372,412],axis=0) #deleting rows manually
# X_mod = np.delete(X,[9,10,11,12,13,14,198],axis=1) #deleting column[if i want to delete only column]
X_mod = np.delete(X,[11, 13, 198, 19,67,69,83,131,132,139,141,143,145,151,156,157,164,204,264,274 ],axis=1) #deleting column 

for i in range(X_mod.shape[1]):
	sm = np.sum(X_mod[:,i])
	if(sm<-9999):
		print("Column with no data: ",i+1," :::: ",sm)

	if(sm==0):
		print("Column with sum = O data: ",i+1," :::: ",sm)
	
	# if(sm<10 and sm>0):
	# 	print("Sum of the Column ",i+1," :::: ",sm)

	
	
for j in range(X_mod.shape[0]):
	sm = np.sum(X_mod[j,:])
	if(sm<-9999):
		print("Row with no data: ",j+1," :::: ",sm)

print("Before Cleaning Data total Attribute: ", X.shape[1])
print("After Cleaning Data total Attribute: ", X_mod.shape[1])
############################## ########################## ##############################

############################## Modifying Y data ##############################

y_mod = np.copy(y)
for k in range(y.shape[0]):
	if(y_mod[k]>1):
		y_mod[k]=0
# print(y)
# print(y_mod) ### if y[k] == 0 ? then it is abnormal signal, and if it is 1 then normal signal ###

############################## ########################## ##############################
print(X_mod[:,149])

np.savetxt('X_final.out', X_mod, fmt='%1.4e')   # X_mod equal sized 
np.savetxt('y_final.out', y_mod, fmt='%1.4e')   # X_mod equal sized 
