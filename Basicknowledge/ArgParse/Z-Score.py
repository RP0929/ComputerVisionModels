import numpy as np

X = np.random.rand(3,3,3)
print(X)
#axis = 0 代表对横轴操作，也就是第0轴；
#axis = 1 代表对纵轴操作，也就是第1轴；
X -= np.mean(X,axis=0)
X /= np.std(X,axis=0)

print(X)
