# demo1  Check verison and install path

import os
import sys

import sklearn
import numpy
import pandas
import matplotlib
import scipy
import tensorflow
import keras

print(f"python executable is:{sys.executable}")
print(f"working directory is:{os.getcwd()}")
print(f"sklearn version:{sklearn.__version__}")
print(f"numpy version:{numpy.__version__}")
print(f"pandas version:{pandas.__version__}")
print(f"matplotlib version:{matplotlib.__version__}")
print(f"scipy version:{scipy.__version__}")
print(f"tensorflow version:{tensorflow.__version__}")
print(f"keras version:{keras.__version__}")