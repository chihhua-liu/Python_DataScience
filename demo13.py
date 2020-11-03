import numpy as np    # view()

a = np.array([[1, 6], [3, 7]])
b = a.view()
print(a)
print(b)
a.shape = (4, -1)
print(a)
print(b)
c = a
c.shape = (1, 4)
print('---')
print(a)
print(b)
print(c)