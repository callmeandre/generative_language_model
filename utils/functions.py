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
    
def add_all_percentiles(perc_list, df_data, ax, drop, y_loc=800, spacer=5):
    for itm in perc_list:
        ax = add_percentile_lines(df_data, ax, itm, y_loc, spacer)
        y_loc = y_loc - drop
        
    return ax
    
def capping_length(val, length):
    if int(val) > length: new_val = length
    else: new_val = val
    return int(new_val)

def post_and_reply_length(df_data, post='parent_body', reply='body', cap=500):
    
    df_data['parent_length'] = df_data[post].apply(lambda x:len(x.split())) 
    df_data['length'] = df_data[reply].apply(lambda x:len(x.split())) 
    
    df_data['parent_length_cap'] = df_data['parent_length'].apply(lambda x: capping_length(x, cap))
    df_data['length_cap'] = df_data['length'].apply(lambda x: capping_length(x, cap))
    
    fig, axs = plt.subplots(2, 1, sharey=True, figsize=(15, 10))

    axs[0].hist(df_data['parent_length_cap'], bins=100, color = "royalblue")
    axs[0].title.set_text('Parent Post Length Capped at %s Words' % (cap))
    axs[0] = add_all_percentiles([50, 75, 90, 95, 99], df_data['parent_length'], axs[0], 50, 800)
    axs[1].hist(df_data['length_cap'], bins=100, color = "peachpuff")
    axs[1].title.set_text('Reply Length Capped at %s Words' % (cap))
    axs[1] = add_all_percentiles([50, 75, 90, 95, 99], df_data['length'], axs[1], 50, 800)
    
    plt.suptitle("Comparing Post Word Count by Type", y=1.03, verticalalignment='top', fontsize = 20)
    plt.tight_layout()
    
    