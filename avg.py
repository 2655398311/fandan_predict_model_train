import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor


def huanyuan_y(y_data):
    # y_data 格式为 list, ；例如 [0.1324513]
    y_data = pd.Series(y_data)
    y_raw = y_data * all_data[y_name].std() + all_data[y_name].mean()
    return y_raw.values.tolist()



def cal_acc(y_predict, y_test):
	# 数据还原
    y_test = pd.Series(y_test)
    y_test_raw = y_test * all_data[y_name].std() + all_data[y_name].mean()
    y_predict = pd.Series(y_predict)
    y_predict = y_predict * all_data[y_name].std() + all_data[y_name].mean()
    
    
#    y_test_raw = [y * all_data[y_name].std() + all_data[y_name].mean() for y in y_test]
#    y_predict = [y * all_data[y_name].std() + all_data[y_name].mean() for y in y_predict]
    # 单个样例的准确率计算公式为： abs(（预测出来的y - 真正的y）/ 真正的y )
    # 所有样例的准确率即等于 各个样例的准确率加总/样例个数
    acc1 = (y_predict - y_test_raw).abs() / y_test_raw
    acc2 = acc1.apply(lambda x: int(x))
    acc = 1 - (acc1 - acc2).sum() / len(y_test)
#    acc = 1 - np.sum([np.abs(y_predict[i] - y_test_raw[i]) / y_test_raw[i] - \
#             int(np.abs(y_predict[i] - y_test_raw[i]) / y_test_raw[i]) for i in range(len(y_test_raw))]) / len(y_test)
    return acc



#def predict(x_to_predict, x, y):
#    for i in range(len(x_to_predict)):
#        x_to_predict[i] = (x_to_predict[i] - x_mean[i]) / x_std[i]
#    
#    # 运行一百次,取平均
#    _predict = []
#    for i in range(100):
#        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
#        xgb6 = xgb.XGBRegressor().fit(X_train, y_train)
#        _predict.append(xgb6.predict(x_to_predict))
#    return [int(float(yi) * y_std + y_mean) for yi in _predict]


start_time = datetime.datetime.now()
data_name = ['avg_view_num', 'avg_vistor_num', 'avg_add_cart_num', 'avg_order_buyers_num',
             'avg_collect_people_num', 'avg_payment_buyers', 'avg_payment_goods_num_l7',
             'avg_stay_time', 'avg_page_bounce_rate', 'avg_order_payment_conversion_rate',
             'avg_payment_conversion', 'avg_order_goods_num', 'avg_per_ticket_sales',
             'avg_vistor_avg_value', 'sum_payment_goods_num_n7']
datum = pd.read_csv('train_datav2.csv', encoding = 'gb2312')
# all_data = datum[data_name]
all_data = datum[data_name].iloc[:80000]
y_name = 'sum_payment_goods_num_n7'
x_name = data_name[:-1]


# 数据标准化

all_data1 = pd.DataFrame()
print(all_data1)
for x in x_name:
    all_data1[x] = (all_data[x] - all_data[x].mean()) / all_data[x].std()
all_data1[y_name] = (all_data[y_name] - all_data[y_name].mean()) / all_data[y_name].std()

x = all_data1[x_name].values.tolist()
y = all_data1[y_name].values.tolist()


# 将数据顺序打乱，每次随机选取一定比例的数据当做测试集（这里设为了30%）
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
bdt6 = BaggingRegressor(DecisionTreeRegressor(criterion='mse'))
bdt6.fit(X_train, y_train)
y_predict = bdt6.predict(X_test)
acc = cal_acc(y_predict, y_test)
print('model1 准确率为：', acc)


from sklearn.externals import joblib
joblib.dump(bdt6, 'bdt6.pkl')


try:
    xgb6 = xgb.XGBRegressor()
except:
    xgb6 = xgb.XGBRegressor()
xgb6.fit(X_train, y_train)
y_predict = xgb6.predict(X_test)
acc = cal_acc(y_predict, y_test)
print('model2 准确率为：', acc)

# 保存
joblib.dump(xgb6, 'xgb6.pkl')


predict_tuple_list = [
	([611497.5714	, 220576.5714, 25866.71429, 3138.428571, 8912.571429, 2904.142857,
   3776.571429, 20.73857143, 0.510785714, 0.894942857, 0.014357143, 4238.571429, 361.6957143, 5.115714286], 
    1.57142857142857),
]
    

for x_to_predict, y_true in predict_tuple_list:
    x_to_predict1 = []
    for i in range(len(x_name)):
        tmp_x_to_predict = (x_to_predict[i] - all_data[x_name[i]].mean()) / all_data[x_name[i]].std()
        x_to_predict1.append(tmp_x_to_predict)
    x_to_predict = [x_to_predict1]
    y_true = [y_true]
    y_true = huanyuan_y(y_true)
    _predict = xgb6.predict(x_to_predict)
    _predict = huanyuan_y(_predict)[0]
    y_predict = np.mean(_predict)
    print('100次结果：', _predict)
    print('平均结果:', y_predict)
    print("finish current.")


end_time = datetime.datetime.now()
print('运行时间:', str((end_time - start_time).seconds) + 's')

all_data_mean = []
all_data_std = []
for i in range(len(x_name)):
    tmp_mean = all_data[x_name[i]].mean()
    tmp_std = all_data[x_name[i]].std()
    all_data_mean.append(tmp_mean)
    all_data_std.append(tmp_std)

all_data_mean.append(all_data[y_name].mean())
all_data_std.append(all_data[y_name].std())

a = pd.DataFrame(all_data_mean,columns=['mean'])
b = pd.DataFrame(all_data_std,columns=['std'])
c = pd.concat([a,b],axis=1)
# print(c)
# cd = pd.DataFrame(all_data_mean,all_data_std,columns=['mean','std'])
# cd.to_csv('std_mean.csv')