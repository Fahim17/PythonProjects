import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

py = [
      [0.931, 0.925, 0.880, 0.907, 0.928, 0.915, 0.918, 0.914],
      [0.854, 0.861, 0.861, 0.861, 0.861, 0.861, 0.865, 0.865],
      [0.941, 0.937, 0.912, 0.908, 0.897, 0.901, 0.904, 0.926],
      [0.949, 0.949, 0.949, 0.949, 0.949, 0.949, 0.949, 0.949],
      [0.903, 0.907, 0.907, 0.907, 0.907, 0.907, 0.907, 0.907]
]





le = ['RBBB', 'CAD', 'MI', "ST", "SB"]


plt.figure(figsize=(10, 8))
px = np.arange(50,450,50)

ct = dict(zip(le, py))
df = pd.DataFrame(data=py, columns=px)
print(df)
# df.to_csv("ANN.csv")

for i in range(5):    
    plt.plot(px, py[i], label=le[i])
plt.legend(loc = 1)
plt.title('Cross Validation Score For ANN')
plt.xlabel('Neuron Count')
plt.ylabel('Score')
#plt.ylim(0, 1) 
plt.show()