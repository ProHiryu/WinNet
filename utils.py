#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2018 - songheqi <songheqi1996@gmail.com>

from crawler import get_game_details, get_game_list

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

import os
import configparser
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

conf = configparser.ConfigParser()

conf.read('conf.ini')
year = conf.getint('config', 'year')
start_month = conf.getint('config', 'start_month')
current_month = conf.getint('config', 'current_month')
title_number = conf.getint('config', 'title_number')
save_path = conf.get('config', 'save_path')
save_file_name = conf.get('config', 'save_file_name')


def download_dataset():
    '''
    download data from wanplus.com
    details in crawler.py
    '''
    dataset = []

    try :
        for month in range(start_month, current_month + 1):
            game_list = get_game_list(year=year, month=month)
            # print(game_list)
            game_series_ids = [int(game['id']) for game in game_list]
            for game_series_id in tqdm(game_series_ids, ascii=True, desc="Downloading month {} of {} data..".format(month, year)):
                game_sets = get_game_details(game_series_id)
                # print(game_sets)
                for game in game_sets:
                    single_game = []
                    for i, team in enumerate(game['participants_teams']):
                        single_game.append(i)
                        single_game.append(team['name'])
                        pre = ''
                        for player in team['players']:
                            if player['title'] == pre:
                                # print(player['title'])
                                continue
                            pre = player['title']
                            single_game.append(player['name'])
                            single_game.append(player['title'])
                        
                        single_game.append(game['participants_teams'][i]['is_win'])
                        dataset.append(single_game)
                        single_game = []
                        # print(dataset)
            print("Download month {} completed!!".format(month))
    except TypeError:
        pass

    df = pd.DataFrame(dataset)
    safe_mkdir(save_path)
    df.to_csv(save_path + save_file_name, index=False)



def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def _one_hot(data, vocab, insert=False):
    '''
    parse data and make one-hot coding
    '''
    vocab_dict = {}
    lenth = len(vocab)
    for i, v in enumerate(vocab):
        vocab_dict[v] = i + 1
    new_data = []
    for row in data:
        new_row = []
        try:
            for item in row:
                new_item = np.array([0] * (vocab_dict[item] - 1) + [1] + [0] * (lenth - vocab_dict[item]))
                new_row.append(new_item)
        except:
            if row == "TRUE":
                new_row = [1, 0]
            else:
                new_row = [0, 1]
        new_data.append(new_row)
    
    if insert:
        data = []
        for row in new_data:
            stack1 = []
            stack2 = []
            new_row = row[0]
            for item in row[1:5]:
                new_row += item
            stack1 = new_row.tolist()
            new_row = row[5]
            for item in row[6:]:
                new_row += item
            stack2 = new_row.tolist()
            data.append(stack1 + stack2)

        new_data = data

    return np.array(new_data, dtype='float32')


def drop_duplicates(row):
    row = np.array(row)
    sets = set()
    new_row = []
    for i in row[1:]:
        if i not in sets:
            new_row.append(i)
            sets.add(i)
        else:
            pass
    return pd.Series(new_row)


def read_data(test_rate=0.3, val_rate=0):
    '''
    read data from file
    '''
    df = pd.read_csv(save_path + save_file_name)
    df = df.apply(drop_duplicates ,axis=1)
    df = df.iloc[:,:11]
    df = df.dropna(axis=0, how='any')
    # print(df)

    x, y = df.iloc[:,:-1], df.iloc[:,-1]
    x = np.array(x).tolist()
    y = np.array(y).tolist()
    # print(x[0], y)

    vocab = list(set([title for row in x for title in row]))
    labels = ['False', 'True']
    X = _one_hot(x, vocab, insert=True)
    Y = _one_hot(y, labels)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_rate, random_state=21)
    # print(X_test.shape)
    return  (X_train, y_train), (X_test, y_test)



def get_dataset(batch_size):
    # Step 0: Download data
    if os.path.exists(save_path + save_file_name):
        print('Already has Data.')
    else:
        download_dataset()

    # Step 1: Read in data
    train, test = read_data()

    # Step 2: Create datasets and iterator
    train_data = tf.data.Dataset.from_tensor_slices(train)
    train_data = train_data.shuffle(10000) # if you want to shuffle your data
    train_data = train_data.batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices(test)
    test_data = test_data.batch(batch_size)

    return train_data, test_data

download_dataset()