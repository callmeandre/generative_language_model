# -*- coding: utf-8 -*-
"""
@authors: Andre Fernandes and Miguel Jaime

Functions for Generative Language Model Project

"""

#####################################
# imports
#####################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, MaxPool2D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import optimizers, backend
from keras.models import load_model
import time
import pickle
import os

#####################################
# functions
#####################################


