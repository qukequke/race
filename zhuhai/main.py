import pandas as pd
import pickle
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import time

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import numpy as np


def fuzhu(x):
    x = int(x[:2])
    if 5 <= x < 10:
        d = '1'
    elif 10 <= x < 15:
        d = '2'
    elif 15 <= x < 20:
        d = '3'
    else:
        d = '0'
    #     elif 45<=x<59:
    #         d = '4'
    return d


def get_test_data(df=None):
    test_df = pd.read_csv('data/test_data.csv')
    return test_df


def get_df_weather():
    df_weather = pd.read_csv('./data/weather.txt', names=['datetim', 'loc', 'low_tem', 'high_tem', 'wea'])
    df_weather['is_rainy'] = df_weather['wea'].str.contains('雨').astype(int)
    df_weather['datetim'] = df_weather['datetim'].astype(str)
    weather_dict = {w: i for i, w in enumerate(df_weather['wea'].unique())}
    df_weather['wea'] = df_weather['wea'].map(weather_dict)
    return df_weather


def get_weekends():
    weekends = {10: [5, 6, 12, 13, 19, 20, 26, 27], 11: [2, 3, 9, 10, 16, 17, 23, 24, 30],
                12: [1, 7, 8, 14, 15, 21, 22, 28, 29]}
    weekends = [f"{month}-{str(day).zfill(2)}" for month, day_list in weekends.items() for day in day_list]
    return weekends


def get_main_df():
    df_weather = get_df_weather()
    weekends = get_weekends()
    df = pd.read_csv("E:/all_project/race\zhuhai/data\网格人流预测.txt", sep='|')
    test_df = get_test_data()
    test_df['datatim'] = test_df['datatim'].astype(str)
    df = pd.concat([df, test_df], axis=0)
    df['datetime'] = df['datatim'].apply(lambda x: x[:8])
    df = df.merge(df_weather, left_on='datetime', right_on='datetim')
    df['datatim'] = df['datatim'].apply(lambda x: f"{x[4:6]},{x[6:8]},{x[8:10]},{x[10:]}")
    df = pd.concat([df['datatim'].str.split(',', expand=True), df], axis=1)
    df = df.rename(columns={0: 'month', 1: 'day', 2: 'hour', 3: 'time', 'cnt': 'flow'})
    df = df[df['month'] != '']

    df['time'] = df['time'].apply(fuzhu)
    df = df[df['time'].isin(['1', '2', '3'])]
    # print(df)
    df['time_label_all'] = df['month'] + '-' + df['day'] + '-' + df['hour'] + '-' + df['time']
    df['time_label'] = df['time_label_all'].apply(lambda x: x[6:])
    df['is_weekend'] = (df['month'] + '-' + df['day']).isin(weekends).astype(int)
    df = df.groupby(
        ['is_rainy', 'time_label', 'grid_id', 'c_long', 'c_lat', 'time_label_all', 'is_weekend', 'day', 'month',
         'low_tem',
         'high_tem', 'wea']).sum()['flow'].reset_index().sort_values(['grid_id', 'month', 'day'])
    df['day'] = df['day'].astype(int)
    df['month'] = df['month'].astype(int)
    df['hour'] = df['time_label'].str.split('-', expand=True)[0].astype(int)
    df = df['hour'] = df[df['hour'] == 0]
    # df_save = df.copy()
    return df


def feature_extraction(flow_data_in):
    flow_data_in['flow_1db'] = [flow_data_in['flow'].mean()] * 1 + flow_data_in['flow'][:-1].tolist()
    flow_data_in['flow_2db'] = [flow_data_in['flow'].mean()] * 2 + flow_data_in['flow'][:-2].tolist()
    flow_data_in['flow_3db'] = [flow_data_in['flow'].mean()] * 3 + flow_data_in['flow'][:-3].tolist()
    flow_data_in['flow_7db'] = [flow_data_in['flow'].mean()] * 7 + flow_data_in['flow'][:-7].tolist()
    #     flow_data_in['flow_14db'] = [flow_data_in['flow'].mean()]*14 + flow_data_in['flow'][:-14].tolist()
    #     flow_data_in['flow_21db'] = [flow_data_in['flow'].mean()]*21 + flow_data_in['flow'][:-21].tolist()
    #     flow_data_in['flow_7_14db'] = flow_data_in[['flow_7db','flow_14db', 'flow_21db']].mean(axis=1)

    flow_data_in['wea_1db'] = [0] * 1 + flow_data_in['wea'][:-1].tolist()
    flow_data_in['is_rainy'] = [0] * 1 + flow_data_in['is_rainy'][:-1].tolist()
    flow_data_in['low_tem_1db'] = [0] * 1 + flow_data_in['low_tem'][:-1].tolist()
    flow_data_in['high_tem_1db'] = [0] * 1 + flow_data_in['high_tem'][:-1].tolist()

    flow_data_in['flow_3dba'] = flow_data_in[['flow_1db', 'flow_2db', 'flow_3db']].mean(axis=1)

    df_grid_flow = flow_data_in.groupby('grid_id').mean().reset_index()[['grid_id', 'flow']]
    df_grid_flow = df_grid_flow.rename(columns={'flow': 'area_flow'})
    flow_data_in = flow_data_in.merge(df_grid_flow, on='grid_id')
    return flow_data_in


def training_xgboost(train_data, train_y):
    model_xgb = xgb.XGBRegressor(max_depth=3
                                 , learning_rate=0.14
                                 , n_estimators=1000
                                 , n_jobs=-1)
    t1 = time.time()
    model_xgb.fit(train_data, train_y)
    print('training time:', str(int(time.time() - t1)) + 's, ', end='')
    return model_xgb


def training_lightgbm(train_data, train_y):
    model_lgb = lgb.LGBMRegressor(num_leaves=20
                                  , max_depth=3
                                  , learning_rate=0.14
                                  , n_estimators=2000
                                  , n_jobs=-1)

    t1 = time.time()
    model_lgb.fit(train_data, train_y)
    print('training time:', str(int(time.time() - t1)) + 's, ', end='')
    return model_lgb


def train_test_split(train_test_df):
    test_train_split_month, test_train_split_day = 12, 30
    test_train_start_month, test_train_start_day = 10, 5
    train_df = train_test_df[((train_test_df['day'].astype(int) > 5) & (train_test_df['month'] == 10) | (
            train_test_df['month'] == 11) | ((train_test_df['month'] == 12) & (
            train_test_df['day'] <= test_train_split_day)))]
    #     test_df = train_test_df[~train_test_df.index.isin(train_df.index)]
    test_df = train_test_df[(train_test_df['day'].astype(int) > test_train_split_day) & (train_test_df['month'] == 12)]
    train_y = train_df.pop('flow')
    train_x = train_df
    test_y = test_df.pop('flow')
    test_x = test_df
    return train_x, train_y, test_x, test_y


def save_model(model, path):
    pickle.dump(model, open(path, "wb"))


def train(df):
    df['month_day'] = df['month'].astype(str) + '-' + df['day'].astype(str)
    result_df = pd.DataFrame()
    drop_columns = ['time_label_all', 'time_label', 'month_day', 'hour']
    hour = 0
    for i in range(1, 4):
        time_label = f"{str(hour).zfill(2)}-{i}"
        train_test_df = df[df['time_label'] == time_label]
        train_test_df.drop(drop_columns, axis=1, inplace=True)
        train_test_df = feature_extraction(train_test_df)
        train_x, train_y, test_x, test_y = train_test_split(train_test_df)
        model = training_xgboost(train_x, train_y)
        pred_y = model.predict(test_x)
        model2 = training_lightgbm(train_x, train_y)
        save_model(model, f'model/xgb_{i}')
        save_model(model2, f'model/gbm_{i}')

        pred_y2 = model2.predict(test_x)

        test_x['pred'] = pred_y
        test_x['pred_2'] = pred_y2
        test_x['flow'] = test_y
        test_x['time_label'] = time_label
        test_x['error'] = abs(test_x['pred'] - test_x['flow'])
        result_df = result_df.append(test_x.copy())

    result_df['merge_pred'] = result_df['pred'] * 0.5 + result_df['pred_2'] * 0.5
    time_label_dict = {'00-1': '20191231000500', '00-2': '20191231001000', '00-3': '20191231001500'}
    result_df = result_df[['grid_id', 'c_long', 'c_lat', 'time_label', 'merge_pred']]
    result_df['time_label'] = result_df['time_label'].apply(lambda x: time_label_dict[x])
    result_df = result_df.rename(columns={'time_label': 'datatim', 'merge_pred': 'cnt'})
    return result_df


def predict():
    df = pd.read_csv('data/input_df.csv')
    df['month_day'] = df['month'].astype(str) + '-' + df['day'].astype(str)
    result_df = pd.DataFrame()
    drop_columns = ['time_label_all', 'time_label', 'month_day', 'hour']
    hour = 0
    for i in range(1, 4):
        print(i)
        time_label = f"{str(hour).zfill(2)}-{i}"
        train_test_df = df[df['time_label'] == time_label]
        train_test_df.drop(drop_columns, axis=1, inplace=True)
        train_test_df = feature_extraction(train_test_df)
        train_x, train_y, test_x, test_y = train_test_split(train_test_df)
        model = pickle.load(open(f"model/xgb_{i}", "rb"))
        pred_y = model.predict(test_x)
        model2 = pickle.load(open(f"model/gbm_{i}", "rb"))

        pred_y2 = model2.predict(test_x)

        test_x['pred'] = pred_y
        test_x['pred_2'] = pred_y2
        test_x['time_label'] = time_label
        result_df = result_df.append(test_x.copy())

    result_df['merge_pred'] = result_df['pred'] * 0.5 + result_df['pred_2'] * 0.5
    time_label_dict = {'00-1': '20191231000500', '00-2': '20191231001000', '00-3': '20191231001500'}
    result_df = result_df[['grid_id', 'c_long', 'c_lat', 'time_label', 'merge_pred']]
    result_df['time_label'] = result_df['time_label'].apply(lambda x: time_label_dict[x])
    result_df = result_df.rename(columns={'time_label': 'datatim', 'merge_pred': 'cnt'})
    return result_df


if __name__ == '__main__':
    mode = 'predict'
    # mode = 'train'
    if mode == 'predict':
        result_df = predict()
        result_df.to_csv('output/result.csv', index=False)
    else:
        df = get_main_df()
        df.to_csv('data/input_df.csv', index=False)
        train(df)
