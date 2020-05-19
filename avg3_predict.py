import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

user = 'root'
passwd = 'Chenfan@123.com.cn.'
host = '10.228.81.237'  ###正式机
port = 30001
dbname1 = 'dm_event'

def huanyuan_y(y_data):
    y_data = pd.Series(y_data)
    y_raw = y_data * all_data[y_name].std() + all_data[y_name].mean()
    return y_raw.values.tolist()
#
#
def cal_acc(y_predict, y_test):
    # 数据还原
    y_test = pd.Series(y_test)
    y_test_raw = y_test * all_data[y_name].std() + all_data[y_name].mean()
    y_predict = pd.Series(y_predict)
    y_predict = y_predict * all_data[y_name].std() + all_data[y_name].mean()
#
#     #    y_test_raw = [y * all_data[y_name].std() + all_data[y_name].mean() for y in y_test]
#     #    y_predict = [y * all_data[y_name].std() + all_data[y_name].mean() for y in y_predict]
#     # 单个样例的准确率计算公式为： abs(（预测出来的y - 真正的y）/ 真正的y )
#     # 所有样例的准确率即等于 各个样例的准确率加总/样例个数
    acc1 = (y_predict - y_test_raw).abs() / y_test_raw
    acc2 = acc1.apply(lambda x: int(x))
    acc = 1 - (acc1 - acc2).sum() / len(y_test)
    return acc
data_name = ['avg_view_num','avg_vistor_num','avg_add_cart_num','avg_order_buyers_num',
             'avg_collect_people_num','avg_payment_buyers','avg_payment_goods_num_l3','avg_stay_time',
             'avg_page_bounce_rate','avg_order_payment_conversion_rate','avg_payment_conversion',
             'avg_order_goods_num','avg_per_ticket_sales','avg_vistor_avg_value','avg_payment_goods_num_n3']
datum = pd.read_csv('train_datav5.csv',encoding='gb2312')
all_data = datum[data_name]
y_name = 'avg_payment_goods_num_n3'
x_name = data_name[:-1]

all_data1 = pd.DataFrame()
for x in x_name:
    all_data1[x] = (all_data[x] - all_data[x].mean()) / all_data[x].std()
all_data1[y_name] = (all_data[y_name] - all_data[y_name].mean()) / all_data[y_name].std()
x = all_data1[x_name].values.tolist()
y = all_data1[y_name].values.tolist()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
bdt6 = BaggingRegressor(DecisionTreeRegressor(criterion='mse'))
bdt6.fit(X_train, y_train)
y_predict = bdt6.predict(X_test)
acc = cal_acc(y_predict, y_test)
print('model1 准确率为：', acc)
from sklearn.externals import joblib
joblib.dump(bdt6, 'bdt6_avg_3.pkl')


xgb6 = xgb.XGBRegressor()
xgb6.fit(X_train, y_train)
y_predict = xgb6.predict(X_test)
acc = cal_acc(y_predict, y_test)
print('model2 准确率为：', acc)

# joblib.dump(xgb6, 'xgb6.pkl')

predict_tuple_list = [
    ([422.3333333,260.3333333,21.33333333,11,2,10.66666667,11.66666667,8.376666667,0.8508,0.9394,0.039633333,13,87.43333333,3.45],3.333333333)]

for x_to_predict, y_true in predict_tuple_list:
    x_to_predict1 = []
    for i in range(len(x_name)):
        tmp_x_to_predict = (x_to_predict[i] - all_data[x_name[i]].mean()) / all_data[x_name[i]].std()
        x_to_predict1.append(tmp_x_to_predict)
    x_to_predict = [x_to_predict1]
    y_true = [y_true]
    _predict = xgb6.predict(x_to_predict)[0]
    _predict = huanyuan_y(_predict)[0]
    y_predict = np.mean(_predict)
    print('100次结果：', _predict)
    print('平均结果:', y_predict)
    print("finish current.")
all_data_mean = []
all_data_std = []
for i in range(len(x_name)):
    tmp_mean = all_data[x_name[i]].mean()
    tmp_std = all_data[x_name[i]].std()
    all_data_mean.append(tmp_mean)
    all_data_std.append(tmp_std)

all_data_mean.append(all_data[y_name].mean())
all_data_std.append(all_data[y_name].std())
mean_a = pd.DataFrame(all_data_mean,columns=['mean'])
std_a = pd.DataFrame(all_data_std,columns=['std'])
mean_std = pd.concat([mean_a,std_a],axis=1,join='inner')
mean_std.to_csv('mean_std_3_avg.csv')
# print(mean_std)
# # 将结果写入到数据库
# engine1 = create_engine("mysql+pymysql://%s:%s@%s:%s/%s?charset=utf8" % (user, passwd, host, port, dbname1))
# # print(engine1)
# pd.io.sql.to_sql(mean_std, 'f_mean_std_3_avg', engine1, schema='dm_event', if_exists='append')