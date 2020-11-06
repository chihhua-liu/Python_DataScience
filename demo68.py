# demo68 : the difference of huge number and small number have calculate  error

y = 100000000


def calculate(x):
    for i in range(0, 1000000):
        x += 0.0000001
    x -= 0.1
    return x


print('%.7f' % calculate(y))