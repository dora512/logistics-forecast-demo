import pandas as pd
import os

# 1. 读取订单表
orders = pd.read_csv('olist_orders_dataset.csv')

# 2. 转换日期格式
orders['order_purchase_date'] = pd.to_datetime(orders['order_purchase_timestamp']).dt.date

# 3. 只保留已送达的订单（可选，也可以保留所有订单）
delivered = orders[orders['order_status'] == 'delivered']

# 4. 按日期统计订单量
daily_orders = delivered.groupby('order_purchase_date').size().reset_index(name='volume')
daily_orders.columns = ['date', 'volume']

# 5. 按日期排序
daily_orders = daily_orders.sort_values('date')

# 6. 补全缺失日期
date_range = pd.date_range(
    start=daily_orders['date'].min(),
    end=daily_orders['date'].max(),
    freq='D'
).date

daily_orders = daily_orders.set_index('date').reindex(date_range).reset_index()
daily_orders.columns = ['date', 'volume']
daily_orders['volume'] = daily_orders['volume'].fillna(0).astype(int)

# 7. 保存
daily_orders.to_csv('daily_orders.csv', index=False)
print(f"✅ 已生成 daily_orders.csv")
print(f"   日期范围: {daily_orders['date'].min()} ~ {daily_orders['date'].max()}")
print(f"   总天数: {len(daily_orders)}")
print(f"   日均订单量: {daily_orders['volume'].mean():.1f}")

# 继续上面的代码
# 8. 计算配送时长
delivered['purchase_date'] = pd.to_datetime(delivered['order_purchase_timestamp'])
delivered['delivered_date'] = pd.to_datetime(delivered['order_delivered_customer_date'])
delivered['delivery_days'] = (delivered['delivered_date'] - delivered['purchase_date']).dt.days

# 9. 过滤异常值（配送时长超出合理范围）
delivered = delivered[(delivered['delivery_days'] >= 0) & (delivered['delivery_days'] <= 60)]

# 10. 按日期统计平均配送时长
daily_delivery = delivered.groupby('order_purchase_date')['delivery_days'].mean().reset_index()
daily_delivery.columns = ['date', 'volume']

# 11. 按日期排序
daily_delivery = daily_delivery.sort_values('date')

# 12. 补全缺失日期
daily_delivery = daily_delivery.set_index('date').reindex(date_range).reset_index()
daily_delivery.columns = ['date', 'volume']
daily_delivery['volume'] = daily_delivery['volume'].round(1)

# 前向填充缺失值
daily_delivery['volume'] = daily_delivery['volume'].fillna(method='ffill')

# 13. 保存
daily_delivery.to_csv('daily_delivery.csv', index=False)
print(f"\n✅ 已生成 daily_delivery.csv")
print(f"   日期范围: {daily_delivery['date'].min()} ~ {daily_delivery['date'].max()}")
print(f"   平均配送时长: {daily_delivery['volume'].mean():.1f}天")