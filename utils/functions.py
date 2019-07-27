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
import sqlite3
import re
from unicodedata import normalize

#####################################
# functions
#####################################

#####
# for data import & preprocessing
#####

def create_connection(db):
    """ connect to a sqlite database
    :param db: database file
    :return: a sqlite db connection object, 
      none if error
    """
    try:
        conn = sqlite3.connect(db)
        return conn
    except Error as e:
        print(e)
 
    return None

def _include_parents(df, db):
    """ for a given sample of posts, 
      make sure their parent post is
      included in the sample, and remove
      any duplicates.
    
    :param df: dataframe with samples
    :param db: a sqlite db connection object
    :return comp_df: a dataframe complete with
      original posts and their parents
    """
    
    unique_parent_ids_in_df = set(df.parent_id.unique())
    
    sample_parents_sql = "SELECT subreddit, ups, downs, score, body, id, name, link_id, parent_id \
      FROM May2015 \
      WHERE subreddit != '' AND body != '' AND id != '' AND \
      id IN ('{}') \
      ;".format("', '".join(unique_parent_ids_in_df))

    sample_parents_df = pd.read_sql(sample_parents_sql, db)

    # since some parents might already be included, drop
    comp_df = pd.concat([df, sample_parents_df]).drop_duplicates().reset_index(drop=True)
    
    return(comp_df)

def _basic_preprocessing(post):
    """ basic text preprocessing 
    
    :param post: a social media post, or 
      other similar text input
    :return pp_post: a preprocessed post,
      with non-ASCII values, special characters,
      leading/trailing whitespace removed,
      normalized characters, and lowercase
    """
    
    pp_post = re.sub(r'[^\w\s\']', '', post)
    pp_post = re.sub(r'[^\x00-\x7F]', '', pp_post)
    pp_post = re.sub(r'\n', '', pp_post)
    pp_post = pp_post.strip().lower()
    pp_post = normalize('NFKD', pp_post).encode('ascii', 'ignore').decode('utf8')
    return(pp_post)

def get_sample(nrows, db):
    """ extract a random sample of rows from
      the reddit comments dataset, including
      comment's parent comment where applicable,
      and with some basic preprocessing
    
    :param nrows: number of rows to sample
    :param table: table we will sample from
    :param db: a database connection object
    :return df: a deduped dataframe including
      sampled posts and their parent
    """
    
    # get total rows & use to compute percentage sample that will get nrows
    # total_rows = pd.read_sql_query("SELECT COUNT(*) FROM May2015 WHERE subreddit != '' AND body != '' AND id != '';", db).iloc[0,0]
    # hardcoded to save a trip to the DB
    sample_percentage = nrows/54000000

    sample_sql = "SELECT subreddit, ups, downs, score, body, id, name, link_id, parent_id \
                  FROM May2015 \
                  WHERE subreddit != '' AND body != '' AND id != '' AND \
                  ABS(CAST(RANDOM() AS REAL))/9223372036854775808 < {} \
                 ;".format(sample_percentage)
    sample_df = pd.read_sql(sample_sql, db)
    
    sample_df['parent_id'] = sample_df['parent_id'].apply(lambda x: re.sub(r't\d_', '', x))

    # make sure dataset include post parents
    complete_df = _include_parents(sample_df, db)
    
    # do some pre-processing
    complete_df['body'] = complete_df['body'].apply(_basic_preprocessing)
    complete_df['parent_id'] = complete_df['parent_id'].apply(lambda x: re.sub(r't\d_', '', x))
    
    return(complete_df)

def remove_duplicates(df, existing_ids):
    """ remove duplicates from subsequent 
      sample chunks
    
    :param df: new dataframe that (presumably)
      contains duplicates
    :param existing_indices: unique identifiers
      already in our sample set.
    :return deduped_df: dataframe with dupes removed
    :return all_unique_ids: list of ids including
      unique ones from passed dataframe
    """

    unique_ids_in_df = set(df.id.unique())
    
    dupe_indices = existing_ids.intersection(unique_ids_in_df)
    non_dupes = ~df.id.isin(dupe_indices)
    deduped_df = df[non_dupes]

    all_unique_ids = existing_ids.union(unique_ids_in_df)
    
    return(deduped_df, all_unique_ids)

#####
# main notebook
#####

def read_and_process_data(path, workDir):
    df_full = pd.read_csv(os.path.join(workDir, "data", path), engine='python')
    df_full = df_full[df_full.id.notnull()]
    df_full.body = df_full.body.str.lower()

    df_parent = df_full[['body', 'id']].drop_duplicates()
    df_parent = df_parent.rename({'body':'parent_body', 'id' : 'parent_id'}, axis='columns')

    df_merge = df_parent.merge(df_full.drop('link_id', axis=1), on='parent_id', how='inner')
    df_merge = df_merge[(df_merge.body != '[deleted]') & (df_merge.parent_body != '[deleted]')]
    
    df_merge['is_popular'] = df_merge['score'].apply(lambda x: is_popular(x, 15))

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
    axs[0] = add_all_percentiles([50, 75, 90, 95, 97.5, 99], df_data['parent_length'], axs[0], 50, 800)
    axs[1].hist(df_data['length_cap'], bins=100, color = "peachpuff")
    axs[1].title.set_text('Reply Length Capped at %s Words' % (cap))
    axs[1] = add_all_percentiles([50, 75, 90, 95, 97.5, 99], df_data['length'], axs[1], 50, 800)
    
    plt.suptitle("Comparing Post Word Count by Type", y=1.03, verticalalignment='top', fontsize = 20)
    plt.tight_layout()
    
def ups_and_downs(df_data, ups='ups', downs='downs', cap=500):
    
    df_data['ups_cap'] = df_data[ups].apply(lambda x: capping_length(x, cap))
    df_data['downs_cap'] = df_data[downs].apply(lambda x: capping_length(x, cap))
    
    fig, axs = plt.subplots(2, 1, sharey=True, figsize=(15, 10))
    
    axs[0].hist(df_data['ups_cap'], bins=100, color = "royalblue")
    axs[0].title.set_text('%s for Post Capped at %s Words' % (str.title(ups),cap))
    axs[0] = add_all_percentiles([50, 75, 90, 95, 97.5, 99], df_data[ups].astype(int), axs[0], 500, 2500, spacer=1)
    axs[1].hist(df_data['downs_cap'], bins=100, color = "peachpuff")
    axs[1].title.set_text('%s for Post Capped at %s Words' % (str.title(downs), cap))
    axs[1] = add_all_percentiles([50, 75, 90, 95, 97.5, 99], df_data[downs].astype(int), axs[1], 500, 2500, spacer=1)
    
    plt.suptitle("Comparing Polarity of Posts", y=1.03, verticalalignment='top', fontsize = 20)
    plt.tight_layout()
    
def is_popular(val, threshold):
    if int(val) > threshold: new_val = 1
    else: new_val = 0
    return int(new_val)

def create_labeled_data(df, text_col='parent_body', max_seq_length=150, predict_feature='is_popular')
  train_text = df[text_col].tolist()
  train_text = [' '.join(t.split()[0:max_seq_length]) for t in train_text]
  train_text = np.array(train_text, dtype=object)[:, np.newaxis]
  train_label = df[predict_feature].tolist()

  return train_text, train_label

