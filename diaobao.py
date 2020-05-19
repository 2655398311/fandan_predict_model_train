import pandas as pd
import pandas
from sklearn.externals import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pre_data = pd.read_csv('710_3.csv',encoding='gb2312')
data = pd.read_csv('mean_std_3_avg.csv','gb2312')
def huanyuan_y(y_data):
    y_raw = y_data * data.loc[14,'std'] + data.loc[14,'mean']
    return y_raw
x1 = pre_data.loc[0,['avg_view_num', 'avg_vistor_num', 'avg_add_cart_num', 'avg_order_buyers_num','avg_collect_people_num', 'avg_payment_buyers', 'avg_payment_goods_num_l7','avg_stay_time', 'avg_page_bounce_rate',
                      'avg_order_payment_conversion_rate','avg_payment_conversion',
                      'avg_order_goods_num', 'avg_per_ticket_sales','avg_vistor_avg_value']].tolist()
# x1 = [611497.5714	,220576.5714,	25866.71429	,3138.428571	,8912.571429	,2904.142857,	3776.571429,	20.73857143,	0.510785714	,0.894942857,	0.014357143	,4238.571429,	361.6957143,	5.115714286]
print(x1)
# print(data)
x_biaozhuanhua = []
for i in range(len(data)-1):
    c = (x1[i] - data.loc[i,'mean']) / data.loc[i,'std']
    x_biaozhuanhua.append(c)
bdt6 = joblib.load(r'D:\diaobao_biaozhuanhua\bdt6_avg_3.pkl')
c = bdt6.predict([x_biaozhuanhua])
c = huanyuan_y(c)
print(c)
