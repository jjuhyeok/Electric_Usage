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

def fillnan(train, test):

    def interpolate_(test_df):
        # https://dacon.io/competitions/official/235736/codeshare/2844?page=1&dtype=recent
        # 에서 제안된 방법으로
        __methods = {
            '기온(C)': 'quadratic',
            '풍속(m/s)':'linear',
            '습도(%)':'quadratic',
            # precipitation : 강수량
            #'강수량(mm)':'quadratic',
        }

        for col, method in __methods.items():
            test_df[col] = test_df[col].interpolate(method=method)
            if method == 'quadratic':
                test_df[col] = test_df[col].interpolate(method='linear')
    #interpolate_(train)
    #interpolate_(test)

    # '풍속(m/s)' 및 '습도(%)' 피처에 대한 NaN 값은 이전 값으로 채워줍니다.
    #train['풍속(m/s)'] = train['풍속(m/s)'].fillna(method='ffill')
    #train['습도(%)'] = train['습도(%)'].fillna(method='ffill')
    #train = train.dropna(subset=['풍속(m/s)'])

    # '강수량(mm)' 피처의 NaN 값은 0으로 채워줍니다.
    train['강수량(mm)'] = train['강수량(mm)'].fillna(0)

    train['일조(hr)'] = train['일조(hr)'].fillna(0)
    train['일사(MJ/m2)'] = train['일사(MJ/m2)'].fillna(0)

    # '풍속(m/s)' 및 '습도(%)' 피처에 대한 NaN 값은 이전 값으로 채워줍니다.
    #test['풍속(m/s)'] = test['풍속(m/s)'].fillna(method='ffill')
    #test['습도(%)'] = test['습도(%)'].fillna(method='ffill')

    # '강수량(mm)' 피처의 NaN 값은 0으로 채워줍니다.
    test['강수량(mm)'] = test['강수량(mm)'].fillna(0)

    test['일조(hr)'] = test['일조(hr)'].fillna(0)
    test['일사(MJ/m2)'] = test['일사(MJ/m2)'].fillna(0)
    
    return train, test

def preprocessing_all(train, test, building):
    info_series = building.set_index('건물번호')['연면적(m2)']
    train['연면적'] = train['건물번호'].map(info_series)
    test['연면적'] = test['건물번호'].map(info_series)
    info_series = building.set_index('건물번호')['냉방면적(m2)']
    train['냉방면적'] = train['건물번호'].map(info_series)
    test['냉방면적'] = test['건물번호'].map(info_series)
    info_series = building.set_index('건물번호')['태양광용량(kW)']
    train['태양광용량'] = train['건물번호'].map(info_series)
    test['태양광용량'] = test['건물번호'].map(info_series)
    info_series = building.set_index('건물번호')['ESS저장용량(kWh)']
    train['ESS저장용량'] = train['건물번호'].map(info_series)
    test['ESS저장용량'] = test['건물번호'].map(info_series)
    info_series = building.set_index('건물번호')['PCS용량(kW)']
    train['PCS용량'] = train['건물번호'].map(info_series)
    test['PCS용량'] = test['건물번호'].map(info_series)
    train = train.replace('-', np.nan)
    test = test.replace('-', np.nan)
    train['태양광용량'] = train['태양광용량'].fillna(0)
    train['ESS저장용량'] = train['ESS저장용량'].fillna(0)
    train['PCS용량'] = train['PCS용량'].fillna(0)
    test['태양광용량'] = test['태양광용량'].fillna(0)
    test['ESS저장용량'] = test['ESS저장용량'].fillna(0)
    test['PCS용량'] = test['PCS용량'].fillna(0)        
    train['태양광용량'] = train['태양광용량'].astype(float)
    train['ESS저장용량'] = train['ESS저장용량'].astype(float)
    train['PCS용량'] = train['PCS용량'].astype(float)
    test['태양광용량'] = test['태양광용량'].astype(float)
    test['ESS저장용량'] = test['ESS저장용량'].astype(float)
    test['PCS용량'] = test['PCS용량'].astype(float)





    ## 위치
    le = LabelEncoder()
    train['location'] = le.fit_transform(train['location'])
    test['location'] = le.transform(test['location'])


    ## 날짜
    cols = ['num', 'date_time', 'temp', 'rainy' ,'wind','hum' , 'sun', 'MJ' ,'power', 'location', '연면적', '냉방면적', '태양광용량', 'ESS저장용량', 'PCS용량']
    train.columns = cols
    train['date_time'] = train['date_time'].astype(str)
    date = pd.to_datetime(train.date_time, format='%Y%m%d %H')
    train['hour'] = date.dt.hour
    train['day'] = date.dt.weekday
    train['month'] = date.dt.month
    train['week'] = date.dt.isocalendar().week
    cols = ['num', 'date_time', 'temp', 'rainy' ,'wind','hum','location','sun', 'MJ' , '연면적', '냉방면적', '태양광용량', 'ESS저장용량', 'PCS용량']
    test.columns = cols
    test['date_time'] = test['date_time'].astype(str)
    date = pd.to_datetime(test.date_time, format='%Y%m%d %H')
    test['hour'] = date.dt.hour
    test['day'] = date.dt.weekday
    test['month'] = date.dt.month
    test['week'] = date.dt.isocalendar().week
    train['week'] = train['week'].astype('int')
    test['week'] = test['week'].astype('int')

    
    ## sin, cos
    train['sin_time'] = np.sin(2*np.pi*train.hour/24)
    train['cos_time'] = np.cos(2*np.pi*train.hour/24)

    test['sin_time'] = np.sin(2*np.pi*test.hour/24)
    test['cos_time'] = np.cos(2*np.pi*test.hour/24)

    train['sin_day'] = np.sin(2*np.pi*train.day/7)
    train['cos_day'] = np.cos(2*np.pi*train.day/7)

    test['sin_day'] = np.sin(2*np.pi*test.day/7)
    test['cos_day'] = np.cos(2*np.pi*test.day/7)


    ## 공휴일 변수 추가
    train['holiday'] = train.apply(lambda x : 0 if x['day']<5 else 1, axis = 1)
    train.loc[('2022-08-15'<=train.date_time)&(train.date_time<'2022-08-16'), 'holiday'] = 1

    test['holiday'] = test.apply(lambda x : 0 if x['day']<5 else 1, axis = 1)
    test.loc[('2022-08-15'<=test.date_time)&(test.date_time<'2022-08-16'), 'holiday'] = 1

    ## 빌딩별/날짜별/시간별 target mean, std
    num_hour_mean = train.groupby(['num', 'hour','day'])['power'].mean()
    num_hour_std = train.groupby(['num', 'hour','day'])['power'].std()
    num_hour_mean = num_hour_mean.reset_index()
    num_hour_std = num_hour_std.reset_index()
    num_hour_mean.rename(columns={'power': 'num_day_hour_mean'}, inplace=True)
    num_hour_std.rename(columns={'power': 'num_day_hour_std'}, inplace=True)
    train = pd.merge(train, num_hour_mean, how='left', on=['num', 'hour','day'])
    train = pd.merge(train, num_hour_std, how='left', on=['num', 'hour','day'])
    test = pd.merge(test, num_hour_mean, how='left', on=['num', 'hour','day'])
    test = pd.merge(test, num_hour_std, how='left', on=['num', 'hour','day'])



    ## 불쾌지수 
    train['THI'] = 9/5*train['temp'] - 0.55*(1-train['hum']/100)*(9/5*train['hum']-26)+32
    test['THI'] = 9/5*test['temp'] - 0.55*(1-test['hum']/100)*(9/5*test['hum']-26)+32



    ## 불쾌지수 그루핑
    def assign_THI_group(val):
        if val < 75:
            return 0
        elif 75 <= val < 100:
            return 1
        else:
            return None  
    train['THI_group'] = train['THI'].apply(assign_THI_group)
    test['THI_group'] = test['THI'].apply(assign_THI_group)




    ## 냉방도일
    def CDH(xs):
        ys = []
        for i in range(len(xs)):
            if i < 11:
                ys.append(np.sum(xs[:(i+1)]-26))
            else:
                ys.append(np.sum(xs[(i-11):(i+1)]-26))
        return np.array(ys)
    cdhs = np.array([])
    for num in range(1,101,1):
        temp = train[train['num'] == num]
        cdh = CDH(temp['temp'].values)
        cdhs = np.concatenate([cdhs, cdh])
    train['CDH'] = cdhs
    cdhs = np.array([])
    for num in range(1,101,1):
        temp = test[test['num'] == num]
        cdh = CDH(temp['temp'].values)
        cdhs = np.concatenate([cdhs, cdh])
    test['CDH'] = cdhs


    
    ## 건물용도별
    features = ['건물기타', '공공', '대학교', '데이터센터', '백화점및아울렛', 
                '병원', '상용', '아파트', '연구소', '지식산업센터', '할인마트', '호텔및리조트']
    for feature in features:
        train[feature] = 0
        test[feature] = 0        
    num_ranges = [(1, 15, '건물기타'), (16, 23, '공공'), (24, 31, '대학교'), 
                  (32, 36, '데이터센터'), (37, 44, '백화점및아울렛'),
                  (45, 52, '병원'), (53, 60, '상용'), (61, 68, '아파트'),
                  (69, 76, '연구소'), (77, 84, '지식산업센터'), 
                  (85, 92, '할인마트'), (93, 100, '호텔및리조트')]
    for start, end, feature in num_ranges:
        train.loc[(train['num'] >= start) & (train['num'] <= end), feature] = 1
        test.loc[(test['num'] >= start) & (test['num'] <= end), feature] = 1



    ## 온도x습도
    train['tem_x_hum'] = train['temp'] * train['hum']
    test['tem_x_hum'] = test['temp'] * test['hum']


    ## 빌딩별 

    ## 빌딩별 출근시간
    def calculate_commute_times(df):
        df['power_diff'] = df.groupby('num')['power'].diff()
        start_hour = df[(df['power_diff'] > 0) & (df['hour'].between(6,10))].groupby('num')['power_diff'].idxmax().reset_index()
        start_hour.columns = ['num', 'hour_idx']
        start_hour['start_hour'] = df.loc[start_hour['hour_idx'], 'hour'].values
        start_hour = start_hour.drop(columns='hour_idx')
        end_hour = df[(df['power_diff'] < 0) & (df['hour'].between(17,22))].groupby('num')['power_diff'].idxmin().reset_index()
        end_hour.columns = ['num', 'hour_idx']
        end_hour['end_hour'] = df.loc[end_hour['hour_idx'], 'hour'].values
        end_hour = end_hour.drop(columns='hour_idx')
        df = df.merge(start_hour, on='num')
        df = df.merge(end_hour, on='num')
        df = df.drop(columns='power_diff')
        return df
    train = calculate_commute_times(train)
    def calculate_commute_period(df):
        start_hour = df.groupby('num')['start_hour'].first().reset_index()
        end_hour = df.groupby('num')['end_hour'].first().reset_index()
        def is_commute_period(row):
            if row['hour'] >= start_hour.loc[row['num'] - 1, 'start_hour'] and row['hour'] <= end_hour.loc[row['num'] - 1, 'end_hour']:
                return 1
            else:
                return 0
        df['commute_period'] = df.apply(is_commute_period, axis=1)
        return df
    train = calculate_commute_period(train)
    num_commute_time_map = train.groupby('num')[['start_hour', 'end_hour']].first().to_dict('index')
    for num, times in num_commute_time_map.items():
        test.loc[test['num'] == num, 'commute_period'] = ((test['hour'] >= times['start_hour']) & (test['hour'] <= times['end_hour'])).astype(int)
    train.drop(['start_hour', 'end_hour'],axis=1,inplace=True)
    test = test.copy()
    test = test.sort_values(by=['num', 'date_time'])
    test = test.reset_index(drop = True)



    ## 체감온도
    def body_temp(val):
        if val < 21:
            return 0
        elif 21 <= val < 25:
            return 1
        elif 25 <= val < 28:
            return 2
        elif 28 <= val < 31:
            return 3
        elif 31 <= val:
            return 4
        else:
            return None
    train['body_temp'] = train['temp'].apply(body_temp)
    test['body_temp'] = test['temp'].apply(body_temp)


    ## 빌딩별 쉬는날?        
    train['low_power_day'] = 0
    # 건물기타
    train.loc[(train['num'].isin([2, 3]) & (train['day'] == 0)) | ((train['num'] == 5) & (train['day'].between(0,3))), 'low_power_day'] = 1
    # 공공
    train.loc[(train['num'].isin([17, 18, 19, 20, 21, 22, 23])) & (train['day'].isin([5, 6])), 'low_power_day'] = 1
    # 대학교
    train.loc[(train['num'].isin([24, 25, 26, 27, 28, 29, 30, 31])) & (train['day'].isin([5, 6])), 'low_power_day'] = 1
    # 병원
    train.loc[(train['num'].isin([45, 50]) & (train['day'] == 5)), 'low_power_day'] = 0.5
    train.loc[(train['num'].isin([45, 50]) & (train['day'] == 6)), 'low_power_day'] = 1
    train.loc[(train['num'].isin([46, 47, 48, 49, 51, 52])) & (train['day'].isin([5, 6])), 'low_power_day'] = 1
    # 상용
    train.loc[((train['num'].isin([53, 55, 57, 58, 59, 60]) & (train['day'].isin([5, 6]))) | ((train['num'] == 54) & (train['day'] == 0))), 'low_power_day'] = 1
    # 연구소
    train.loc[(train['num'].isin([69, 70, 71, 72, 73, 74, 76]) & (train['day'].isin([5, 6]))), 'low_power_day'] = 1
    # 지식산업센터
    train.loc[(train['num'].isin([77, 78, 79, 80, 82, 83, 84])) & (train['day'].isin([5, 6])), 'low_power_day'] = 1
    # 일단 모든 값을 0으로 초기화
    test['low_power_day'] = 0
    # 건물기타
    test.loc[(test['num'].isin([2, 3]) & (test['day'] == 0)) | ((test['num'] == 5) & (test['day'].between(0,3))), 'low_power_day'] = 1
    # 공공
    test.loc[(test['num'].isin([17, 18, 19, 20, 21, 22, 23])) & (test['day'].isin([5, 6])), 'low_power_day'] = 1
    # 대학교
    test.loc[(test['num'].isin([24, 25, 26, 27, 28, 29, 30, 31])) & (test['day'].isin([5, 6])), 'low_power_day'] = 1
    # 병원
    test.loc[(test['num'].isin([45, 50]) & (test['day'] == 5)), 'low_power_day'] = 0.5
    test.loc[(test['num'].isin([45, 50]) & (test['day'] == 6)), 'low_power_day'] = 1
    test.loc[(test['num'].isin([46, 47, 48, 49, 51, 52])) & (test['day'].isin([5, 6])), 'low_power_day'] = 1
    # 상용
    test.loc[((test['num'].isin([53, 55, 57, 58, 59, 60]) & (test['day'].isin([5, 6]))) | ((test['num'] == 54) & (test['day'] == 0))), 'low_power_day'] = 1
    # 연구소
    test.loc[(test['num'].isin([69, 70, 71, 72, 73, 74, 76]) & (test['day'].isin([5, 6]))), 'low_power_day'] = 1
    # 지식산업센터
    test.loc[(test['num'].isin([77, 78, 79, 80, 82, 83, 84])) & (test['day'].isin([5, 6])), 'low_power_day'] = 1




    ## 전력 사용량의 차이 비율
    power_diff_ratio_dict = {}
    for num in range(1, 101): 
        low_power_day_mean_power = train[(train['num'] == num) & (train['low_power_day'] == 1)]['power'].mean()
        not_low_power_day_mean_power = train[(train['num'] == num) & (train['low_power_day'] == 0)]['power'].mean()
        ambiguous_power_day_mean_power = train[(train['num'] == num) & (train['low_power_day'] == 0.5)]['power'].mean()
        
        if pd.isnull(low_power_day_mean_power) or pd.isnull(not_low_power_day_mean_power):
            power_diff_ratio = 0
        else:
            low_power_day_mean_power = (low_power_day_mean_power + ambiguous_power_day_mean_power / 2) if not pd.isnull(ambiguous_power_day_mean_power) else low_power_day_mean_power
            not_low_power_day_mean_power = (not_low_power_day_mean_power + ambiguous_power_day_mean_power / 2) if not pd.isnull(ambiguous_power_day_mean_power) else not_low_power_day_mean_power
            power_diff_ratio = abs(low_power_day_mean_power - not_low_power_day_mean_power) / not_low_power_day_mean_power        
        power_diff_ratio_dict[num] = power_diff_ratio
    train['power_diff_ratio'] = train['num'].map(power_diff_ratio_dict)
    test['power_diff_ratio'] = test['num'].map(power_diff_ratio_dict)



    # 32번, 33번만 해당하는 피처
    train['power_increase_summer'] = 0
    test['power_increase_summer'] = 0
    # 7, 8월에 해당하는 32, 33번 num에 대해 'power_increase_summer' 값을 1로 설정
    train.loc[(train['num'].isin([32, 33])) & (train['month'].isin([7, 8])), 'power_increase_summer'] = 1
    test.loc[(test['num'].isin([32, 33])) & (test['month'].isin([7, 8])), 'power_increase_summer'] = 1



    train.drop(['sun', 'MJ'],axis=1,inplace=True)
    test.drop(['sun', 'MJ'],axis=1,inplace=True)

    return train, test
