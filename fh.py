import pandas as pd
data = pd.read_csv('train_datav5.csv',encoding='gb2312')
# data['avg_payment_goods_num_n3'].dropna(how='10')
print(data)
