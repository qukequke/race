from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn import svm, tree
from sklearn import feature_selection



def age_graph(df):
    # print(df.info())
    #先画图，画生存者和死亡者的年龄密度， 再给Age分成3类
    df.Age[df.Survived==1].plot(kind='kde')
    df.Age[df.Survived==0].plot(kind='kde')
    plt.legend(('alive', 'die'))
    plt.show()


#df.Age 处理一下
def dis_age(df):
    df.loc[df.Age < 18, 'Age'] = 1
    df.loc[((df.Age >= 18) & (df.Age < 32)), 'Age'] = 2
    df.loc[pd.isnull(df.Age), 'Age'] = 2
    df.loc[df.Age >= 32, 'Age'] = 3
    return df


def increase_fea(df):
    df['Family_size'] = df['SibSp'] + df['Parch'] + 1
    df['Is_alone'] = 1
    df.loc[df['Family_size'] > 1, 'Is_alone'] = 0
    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    return df


def scalar_data(series_name, df):#归一化数据
    # print(df.info())
    scalr_model = preprocessing.StandardScaler()
    fare_scale_para = scalr_model.fit(df[series_name].as_matrix().reshape(-1, 1))
    df[series_name+'_scale'] = scalr_model.fit_transform(df.Fare.as_matrix().reshape(-1, 1), fare_scale_para)
    return df


def dis_miss_fare(df):
    df.loc[pd.isnull(df.Fare), 'Fare'] = df.Fare.mean()
    return df


def dummie(df):
    #吧所有数据都变成0， 1
    # print(len(df.columns))
    dummie_pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')
    dummie_Sex = pd.get_dummies(df['Sex'], prefix='Sex')
    dummie_Family_size = pd.get_dummies(df['Family_size'], prefix='Family_size')
    dummie_Age = pd.get_dummies(df['Age'], prefix='Age')
    # print(len(df.columns))
    if len(df.columns) == 16:
        data_df = pd.concat([df.Survived, df.Is_alone, dummie_Family_size, dummie_pclass, dummie_Sex, df['Fare_scale'], dummie_Age], axis=1)
        #试一下这是用年龄和性别
        # data_df = pd.concat([df.Survived, dummie_pclass, dummie_Sex, dummie_Age], axis=1)
    elif len(df.columns) == 15:
        data_df = pd.concat([df.Is_alone, dummie_Family_size, dummie_pclass, dummie_Sex, df['Fare_scale'], dummie_Age], axis=1)
        # data_df = pd.concat([dummie_pclass, dummie_Sex, dummie_Age], axis=1)
    return data_df


def split_data(data_df):
    train_split_df, test_split_df = train_test_split(data_df, test_size=0.3)
    return train_split_df, test_split_df


def cv(train_split_df, test_split_df, model_list):
    # print(train_split_df.columns)
    for model in model_list:
        model.fit(train_split_df.as_matrix()[:, 1:], train_split_df.as_matrix()[:, 0])
        pre_y = model.predict(test_split_df.as_matrix()[:, 1:])
        print(model.__class__.__name__)
        print(classification_report(test_split_df.Survived.as_matrix().reshape(-1, 1), pre_y))
    return model_list


def full_train(df, model_list):
    for model in model_list:
        model.fit(df.as_matrix()[:, 1:], df.as_matrix()[:, 0])
    return model_list


def predict(df, model_list):
    # print(df.columns)
    pre_y_list = []
    for model in model_list:
        pre_y = model.predict(df.as_matrix())
        pre_y_list.append(pre_y)
    return pre_y_list


def create_model():
    xgb = XGBClassifier(learning_rate=0.001, n_estimators=10000)
    logistic_model = LogisticRegression('l2', tol=1e-7)
    svm_model = svm.SVC()
    tree_model = tree.DecisionTreeClassifier()
    model = [xgb, logistic_model, svm_model, tree_model]

    return model


def save_to_csv(pre_list, test_df):
    for i in range(len(pre_list)):
        ret = pd.DataFrame({'PassengerId': test_df.PassengerId.values, 'Survived':pre_list[i].astype('int32')})
        ret.to_csv('./ret'+str(i)+'.csv', index=False)


def feat_sel(train_new_df, model):
    rfe = feature_selection.RFECV(model, scoring='accuracy')
    rfe.fit(train_new_df.as_matrix()[:, 1:], train_new_df.as_matrix()[:, 0])
    X_rfe = train_new_df.columns.values[1:][rfe.get_support()]
    return X_rfe


def hy_para_sel(df):
    grid = {'learning_rate': [0.1, 0.01, 0.2, 0.001],
            'n_estimators': [100, 10, 1000, 10000]}
    tune_model = model_selection.GridSearchCV(XGBClassifier(),grid, scoring='roc_auc')
    tune_model.fit(df.as_matrix()[:, 1:], df.as_matrix()[:, 0])
    print(tune_model.cv_results_)
    print(tune_model.best_params_)
    return tune_model


def main():
    train_df = pd.read_csv('data/train.csv') #读取数据
    test_df = pd.read_csv('data/test.csv')
    df_raw_list = [train_df, test_df]
    df_new_list = []
    for df in df_raw_list: #测试集和训练集数据一起处理
        df = dis_age(df)
        df = dis_miss_fare(df)
        df = increase_fea(df)
        df = scalar_data('Fare', df)
        df = dummie(df)
        df_new_list.append(df)


    train_new_df, test_new_df = df_new_list #恢复出测试集和训练集
    train_split_df, test_split_df = split_data(train_new_df)
    model_list = create_model()

    # hy_para_sel(train_new_df)
    cross_ret = model_selection.cross_validate(model_list[0], train_new_df.as_matrix()[:, 1:], train_new_df.as_matrix()[:, 0])
    print(cross_ret)
    X_rfe = feat_sel(train_new_df, model_list[0])
    cross_ret1 = model_selection.cross_validate(model_list[0], train_new_df[X_rfe], train_new_df['Survived'])

    # model_list = cv(train_split_df, test_split_df, model_list)
    # print(pd.DataFrame({'PassengerId': list(data_df.columns)[1:], 'coef': list(logistic_model.coef_[0])}))
    model_list = full_train(train_new_df, model_list)# 全部拿去fit模型，然后去预测

    test_pre_y_list = predict(test_new_df, model_list)
    save_to_csv(test_pre_y_list, test_df)


if __name__ == '__main__':
    main()

