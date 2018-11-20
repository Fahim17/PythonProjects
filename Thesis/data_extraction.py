import numpy as np

X = np.loadtxt("X_main.dat")
y = np.loadtxt("y_main.dat")

# ##PQRST, milivolt, all nodes
# column = [
# 		160,161,162,165,166, 
# 		170,171,172,175,176, 
# 		180,181,182,185,186, 
# 		190,191,192,195,196,
# 		200,201,202,205,206,
# 		210,211,212,215,216,
# 		220,221,222,225,226,
# 		230,231,232,235,236,
# 		240,241,242,245,246,
# 		250,251,252,255,256,
# 		260,261,262,265,266,
# 		270,271,272,275,276,
# 		]

# ## QRS, milivolt, all nodes
# column = [
# 		160,161,162, 
# 		170,171,172, 
# 		180,181,182, 
# 		190,191,192,
# 		200,201,202,
# 		210,211,212,
# 		220,221,222,
# 		230,231,232,
# 		240,241,242,
# 		250,251,252,
# 		260,261,262,
# 		270,271,272,
# 		]
# ## QRS, duration msec, QRS milivolt, all nodes
# column = [1,14,
# 		15, 16, 17, 
# 		27, 28, 29, 
# 		39, 40, 41, 
# 		51, 52, 53, 
# 		63, 64, 65, 
# 		75, 76, 77, 
# 		87, 88, 89, 
# 		99, 100, 101, 
# 		111, 112, 113, 
# 		123, 124, 125, 
# 		135, 136, 137, 
# 		147, 148, 149,
# 		160,161,162, 
# 		170,171,172, 
# 		180,181,182, 
# 		190,191,192,
# 		200,201,202,
# 		210,211,212,
# 		220,221,222,
# 		230,231,232,
# 		240,241,242,
# 		250,251,252,
# 		260,261,262,
# 		270,271,272,
# 		]
# ## QS, duration msec, QST milivolt,  HRate, QRS PR QT T P interval, ragged T wave, diphasic T, all 12 nodes
column = [14,4,5,6,7,8,
			15, 17, 25, 26, 
			27, 29, 37, 38, 
			39, 41, 49, 50, 
			51, 53, 61, 62, 
			63, 65, 73, 74, 
			75, 77, 85, 86, 
			87, 89, 97, 98, 
			99, 101, 109, 110, 
			111, 113, 121, 122, 
			123, 125, 133, 134, 
			135, 137, 145, 146, 
			147, 149, 157, 158,
			160, 162, 166, 
			170, 172, 176, 
			180, 182, 186, 
			190, 192, 196, 
			200, 202, 206, 
			210, 212, 216, 
			220, 222, 226, 
			230, 232, 236, 
			240, 242, 246, 
			250, 252, 256, 
			260, 262, 266, 
			270, 272, 276
		]


x_temp = np.array(X[:,column])

row_del = []
# print(x_temp.shape)
# print(y.shape)
# print(X[:,160])
disease = 5

c=0
for i in y:
    if(i!=1 and i!=3 and i!=4):
       row_del.append(c)
    c+=1

# print(row_del)       
x_temp = np.delete(x_temp, row_del, 0)
y = np.delete(y, row_del, 0)
print(x_temp.shape)
print(y)

for k in range(y.shape[0]):
	if(y[k]>1):
		y[k]=0
print(x_temp.shape)
print(y.shape)
print(y)



np.savetxt('data_ex/X_all_pqrst.dat', x_temp, fmt='%1.4e')
np.savetxt('data_ex/y_all_pqrst.dat', y, fmt='%1.4e')     


# np.savetxt('data_ex/X_12_HrQRS_TB.dat', x_temp, fmt='%1.4e')
# np.savetxt('data_ex/y_12_HrQRS_TB.dat', y, fmt='%1.4e')     
