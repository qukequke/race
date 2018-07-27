import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import model_selection

def read_data():
    return pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv')



if __name__ == '__main__':
    origin_data_train, origin_data_test = read_data()
    origin_data_list = [origin_data_train, origin_data_test]

    data_train = origin_data_train.copy()
    # print(data_train['Name'])
    data_train['name_sex'] = data_train['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    print(data_train['name_sex'].value_counts())
    female_list = ['Miss', 'Mrs', 'lady', 'Mile']
    constant_list = ['Mr', 'Mrs', 'Miss', 'Master', 'Sir', 'Ms']
    unconstant_list = ['Dr', 'Rev', 'Col', 'Major', 'Sir', 'Jonkheer', 'Don', 'Mile', 'Capt','Mme']
    print(data_train['name_sex'].isin(unconstant_list))
    print(data_train.loc[data_train['name_sex'].isin(unconstant_list), ['name_sex', 'Sex']])
    # print(data_train.loc[((data_train['name_sex'] != 'Mr') & (data_train['name_sex' != 'Mrs'])), ['Sex']])
    print(data_train['name_sex'].value_counts())
    # print(data_train['name_sex'].value_counts())
    sex_null_passengerId = data_train.loc[pd.isnull(data_train['Sex']), 'PassengerId']
    print(data_train['Sex'].value_counts())
    data_train.loc[sex_null_passengerId, 'Sex'] = 'male'
    print(data_train['Sex'].value_counts())
    data_train.loc[data_train['name_sex'].isin(female_list), 'Sex'] = 'female'
    print(data_train['Sex'].value_counts())
    # data_train.loc[null_index, ]

