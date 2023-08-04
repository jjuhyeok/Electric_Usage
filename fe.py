import sys
import sktime
import tqdm as tq
import xgboost as xgb
import matplotlib
import seaborn as sns
import sklearn as skl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils.plotting import plot_series
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder


def new_fe(train, test):
    train['증가'] = 0
    train['감소'] = 0
    for building_num, group in train.groupby('num'):
        prev_power = group['power'].shift(1)
        increase = (group['power'] > prev_power).astype(int)
        decrease = (group['power'] < prev_power).astype(int)
        train.loc[group.index, '증가'] = increase
        train.loc[group.index, '감소'] = decrease
    summary_data = train.groupby(['num', 'day', 'hour'])[['증가', '감소']].sum().reset_index()
    summary_data.rename(columns={'증가': '증가_day_hour_sum', '감소': '감소_day_hour_sum'}, inplace=True)
    train = pd.merge(train, summary_data, left_on=['num', 'day', 'hour'], right_on=['num', 'day', 'hour'], suffixes=('', '_day_hour_sum'))
    test = pd.merge(test, summary_data, left_on=['num', 'day', 'hour'], right_on=['num', 'day', 'hour'], suffixes=('', '_day_hour_sum'))
    train.drop(['증가','감소'], axis=1, inplace=True)
    train['increase_all'] = 0
    train['decrease_all'] = 0
    train.loc[train['증가_day_hour_sum'] == 0, 'decrease_all'] = 1
    train.loc[train['감소_day_hour_sum'] == 0, 'increase_all'] = 1
    test['increase_all'] = 0
    test['decrease_all'] = 0
    test.loc[test['증가_day_hour_sum'] == 0, 'decrease_all'] = 1
    test.loc[test['감소_day_hour_sum'] == 0, 'increase_all'] = 1

    return train, test
