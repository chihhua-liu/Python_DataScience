import numpy as np

a = np.array([[1, 2], [3, 4]])
b = a
c = a.view()
d = a.copy()
print(a)
print(b)
print(c)
print(d)
print('a[0][0] = 100----')
a[0][0] = 100
print(a)
print(b)
print(c)
print(d)
a.shape = (4,)
print('a.shape = (4,)----')
print(a)
print(b)
print(c)
print(d)
