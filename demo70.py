# demo70 change number to one-hit vectors

from keras import utils

origs = [0, 1, 2, 3, 4]
NUM_DIGIT = 15
for o in origs:
    co = utils.to_categorical(o, NUM_DIGIT)
    print(f"{o}==>{co} with digit{NUM_DIGIT}")