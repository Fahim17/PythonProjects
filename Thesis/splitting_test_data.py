import numpy as np
import random

X = np.loadtxt("X_main.dat")
y = np.loadtxt("y_main.dat")
#01             Normal				          245
#02             Ischemic changes (Coronary Artery Disease)   44
#03             Old Anterior Myocardial Infarction           15
#04             Old Inferior Myocardial Infarction           15
#05             Sinus tachycardy			           13
#06             Sinus bradycardy			           25
#10             Right bundle branch block		           50
disease = 10
testD = 10

#column = [
#		16, 17, 18, 
#		28, 29, 30, 
#		40, 41, 42, 
#		52, 53, 54, 
#		64, 65, 66, 
#		76, 77, 78, 
#		88, 89, 90, 
#		100, 101, 102, 
#		112, 113, 114, 
#		124, 125, 126, 
#		136, 137, 138, 
#		148, 149, 150,
#		160,161,162, 
#		170,171,172, 
#		180,181,182, 
#		190,191,192,
#		200,201,202,
#		210,211,212,
#		220,221,222,
#		230,231,232,
#		240,241,242,
#		250,251,252,
#		260,261,262,
#		270,271,272,
#		]

# x_temp = np.array(X[:,column])
# testSampleX = np.ones([len(column)])
x_temp = np.array(X[:,:])
testSampleX = np.ones(279)
testSampley = []


row_del = []
c=0
for i in y:
   # if(i!=1 and i!=3 and i!=4): #only for MI
    if(i!=1 and i!=disease):
       row_del.append(c)
    c+=1

# print(row_del)       
x_temp = np.delete(x_temp, row_del, 0)
y = np.delete(y, row_del, 0)
# print(x_temp.shape)
# print(x_temp)
# print(y.shape)
for k in range(y.shape[0]):
	if(y[k]>1):
		y[k]=0


t = 0
while(t<testD):
	rnd = random.randint(1,x_temp.shape[0])
	if(y[rnd]==1):
		testSampleX = np.vstack((testSampleX,x_temp[rnd]))
		testSampley.append(1)
		x_temp = np.delete(x_temp, rnd, 0)
		y = np.delete(y, rnd, 0)
		t+=1
t=0		
while(t<testD):
	rnd = random.randint(1,x_temp.shape[0])
	if(y[rnd]==0):
		testSampleX = np.vstack((testSampleX,x_temp[rnd]))
		testSampley.append(0)
		x_temp = np.delete(x_temp, rnd, 0)
		y = np.delete(y, rnd, 0)
		t+=1
testSampleX = np.delete(testSampleX, 0, 0)
print("testSample: ",testSampleX.shape)
print("disease: ",disease)
#print("test sample: ",testD*2)

np.savetxt('test/X_train.dat', x_temp, fmt='%1.4e')    
np.savetxt('test/y_train.dat', y, fmt='%1.4e')
np.savetxt('test/X_test.dat', testSampleX, fmt='%1.4e')    
np.savetxt('test/y_test.dat', testSampley, fmt='%1.4e')
np.savetxt('newAlgo_4diseases/X_test.dat', testSampleX, fmt='%1.4e')    
np.savetxt('newAlgo_4diseases/y_test.dat', testSampley, fmt='%1.4e')