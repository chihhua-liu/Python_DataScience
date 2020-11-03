#demo9

from sklearn import datasets
from pprint import pprint
import matplotlib.pyplot as plt

data1 = datasets.make_regression(10, 6, noise=10)
# x,y,z,p,q,r
dataAll = data1[0]
print(dataAll)

r1 = sorted(dataAll, key=lambda tuple: tuple[0])
r2 = sorted(dataAll, key=lambda tuple: tuple[1])
r6 = sorted(dataAll, key=lambda tuple: tuple[5])
pprint(r1)
pprint(r2)
pprint(r6)

features = data1[0]
labels = data1[1]
for i in range(6):
    plt.scatter(features[:, i], labels)
    plt.show()