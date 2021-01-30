import matplotlib.pyplot as plt
import numpy as np
f = open('tmp.txt','r')
a  = np.array([x.split() for x in f])
plt.plot(a[:,1],a[:,0])
plt.show()
print(a.shape)
