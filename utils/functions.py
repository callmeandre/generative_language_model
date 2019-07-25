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

def read_and_process_data(path, workDir):
    df_full = pd.read_csv(os.path.join(workDir, "data", path))
    df_full = df_full[df_full.name.notnull()]
    df_full.body = df_full.body.str.lower()

    df_parent = df_full[['body', 'name']].drop_duplicates()
    df_parent = df_parent.rename({'body':'parent_body', 'name' : 'parent_id'}, axis='columns')

    df_merge = df_parent.merge(df_full.drop('link_id', axis=1), on='parent_id', how='inner')
    df_merge = df_merge[(df_merge.body != '[deleted]') & (df_merge.parent_body != '[deleted]')]

    return df_merge
    
def add_percentile_lines(df_data, ax, percentile, y_loc=800, spacer=5):
    val = np.percentile(df_data, percentile)
    ax.axvline(val, color='k', linestyle='dashed', linewidth=1)
    ax.text(x=val+spacer, y=y_loc, fontsize=12,s='%sperc: %s' % (percentile, round(val).astype(int)))
    return ax
    
    