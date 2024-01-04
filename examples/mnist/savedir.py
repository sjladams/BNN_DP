import time
import sys

LINUX=False

if LINUX:
    DATA = "data/"
    PLOTS = "plots/"
    TESTS = "tests/"+str(time.strftime('%Y-%m-%d'))+"/"
else:
    DATA = "examples/mnist/data/"
    PLOTS = "examples/mnist/plots/"
    TESTS = "examples/mnist/tests/"+str(time.strftime('%Y-%m-%d'))+"/"

