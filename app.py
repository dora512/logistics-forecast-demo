"""
物流货量预测交互式Demo | Streamlit + XGBoost
运行命令：streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb

# 页面配置
st.set_page_config(
    page_title="物流货量预测Demo",
    page_icon="📦",
    layout="wide"
)

# 标题
st.title("📦 快递网点货量预测 Demo")
st.markdown("> 基于 XGBoost + 滚动预测 | 适用于物流/即时配送场景")
st.markdown("---")

# 侧边栏：参数配置
st.sidebar.header("⚙️ 参数配置")

n_days = st.sidebar.slider("预测天数", 1, 14, 7)
show_details = st.sidebar.checkbox("显示详细结果", value=True)
show_feature_importance = st.sidebar.checkbox("显示特征重要性", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 关于")
st.sidebar.markdown("""
- 数据：模拟某网点历史货量（含趋势、星期、节假日）
- 模型：XGBoost
- 策略：滚动预测（逐日递推）
- 代码：[GitHub仓库](https://github.com/your-username/logistics-forecast-demo)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("👤 肖元 | 数量经济学博士 | 求职：算法/数据分析")


# ============================================
# 1. 生成模拟数据（可替换为真实数据）
# ============================================

@st.cache_data
def generate_data():
    """生成模拟货量数据"""
    np.random.seed(42)

    # 3年数据
    dates = pd.date_range(start='2022-01-01', periods=1095, freq='D')
    n = len(dates)

    # 趋势：年增长10%
    trend = 1000 * (1 + 0.0003 * np.arange(n))

    # 星期效应：周末低20%，周三高10%
    weekday_factor = np.ones(n)
    for i, date in enumerate(dates):
        dow = date.dayofweek
        if dow >= 5:
            weekday_factor[i] = 0.8
        elif dow == 2:
            weekday_factor[i] = 1.1

    # 节假日效应
    holiday_factor = np.ones(n)
    for i, date in enumerate(dates):
        if date.month == 1 and date.day <= 10:
            holiday_factor[i] = 1.3
        elif date.month == 10 and 20 <= date.day <= 30:
            holiday_factor[i] = 1.5
        elif date.month == 11 and 1 <= date.day <= 15:
            holiday_factor[i] = 1.8

    # 随机噪声
    noise = np.random.normal(1, 0.15, n)

    volume = trend * weekday_factor * holiday_factor * noise
    volume = volume.astype(int)

    df = pd.DataFrame({'date': dates, 'volume': volume})
    df.set_index('date', inplace=True)
    return df


# ============================================
# 2. 特征工程
# ============================================

def create_features(df, target_col='volume'):
    """
    构造时序特征（自动适应数据量）
    """
    data = df.copy()

    # 1. 时间特征（始终可用）
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    data['quarter'] = data.index.quarter
    data['dayofyear'] = data.index.dayofyear
    data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)
    data['is_month_start'] = data.index.is_month_start.astype(int)
    data['is_month_end'] = data.index.is_month_end.astype(int)

    n_rows = len(data)

    # 2. 滞后特征（根据数据量动态调整）
    lag_windows = [1, 3, 7]  # 基础窗口
    if n_rows > 30:
        lag_windows.append(14)
    if n_rows > 60:
        lag_windows.append(28)
    if n_rows > 120:
        lag_windows.append(56)

    for w in lag_windows:
        if n_rows > w:
            data[f'lag_{w}'] = data[target_col].shift(w)

    # 3. 滑动窗口特征（根据数据量动态调整）
    roll_windows = [3, 7]  # 基础窗口
    if n_rows > 30:
        roll_windows.append(14)
    if n_rows > 60:
        roll_windows.append(28)

    for w in roll_windows:
        if n_rows > w:
            data[f'rolling_mean_{w}'] = data[target_col].shift(1).rolling(w).mean()
            data[f'rolling_std_{w}'] = data[target_col].shift(1).rolling(w).std()
            data[f'rolling_max_{w}'] = data[target_col].shift(1).rolling(w).max()
            data[f'rolling_min_{w}'] = data[target_col].shift(1).rolling(w).min()

    # 4. 时间衰减特征（安全版本：只使用存在的列）
    if 'lag_1' in data.columns:
        weighted_sum = 0.5 * data['lag_1'].fillna(0)
        if 'lag_2' in data.columns:
            weighted_sum += 0.25 * data['lag_2'].fillna(0)
        if 'lag_3' in data.columns:
            weighted_sum += 0.125 * data['lag_3'].fillna(0)
        if 'lag_4' in data.columns:
            weighted_sum += 0.0625 * data['lag_4'].fillna(0)
        data['weighted_avg_7d'] = weighted_sum
    else:
        data['weighted_avg_7d'] = 0

    # 5. 删除空值
    data = data.dropna()

    return data

# ============================================
# 3. 训练模型
# ============================================

@st.cache_resource
def train_model():
    """训练XGBoost模型"""
    df = generate_data()
    df_feat = create_features(df)

    feature_cols = [col for col in df_feat.columns if col != 'volume']
    X = df_feat[feature_cols]
    y = df_feat['volume']

    # 时间序列划分
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train, verbose=False)

    # 预测与评估
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    return model, feature_cols, df_feat, X_test, y_test, y_pred, mae, mape


# ============================================
# 4. 滚动预测
# ============================================

def rolling_forecast(model, last_data, feature_cols, n_days=7):
    """滚动预测未来n天"""
    predictions = []
    current_data = last_data.copy()

    for i in range(n_days):
        X_pred = current_data[feature_cols].iloc[-1:].copy()
        pred = model.predict(X_pred)[0]
        predictions.append(pred)

        # 构造下一行
        next_date = current_data.index[-1] + timedelta(days=1)
        new_row = {}

        # 时间特征
        new_row['year'] = next_date.year
        new_row['month'] = next_date.month
        new_row['day'] = next_date.day
        new_row['dayofweek'] = next_date.dayofweek
        new_row['quarter'] = next_date.quarter
        new_row['dayofyear'] = next_date.dayofyear
        new_row['is_weekend'] = 1 if next_date.dayofweek >= 5 else 0
        new_row['is_month_start'] = 1 if next_date.is_month_start else 0
        new_row['is_month_end'] = 1 if next_date.is_month_end else 0

        # 滞后特征
        for w in [1, 3, 7, 14, 28, 56]:
            if len(current_data) >= w:
                new_row[f'lag_{w}'] = current_data['volume'].iloc[-w]
            else:
                new_row[f'lag_{w}'] = current_data['volume'].iloc[0]

        # 滑动窗口
        last_volume = current_data['volume']
        for w in [3, 7, 14, 28]:
            window = last_volume.iloc[-min(w, len(last_volume)):]
            new_row[f'rolling_mean_{w}'] = window.mean()
            new_row[f'rolling_std_{w}'] = window.std()

        new_row['weighted_avg_7d'] = (
                0.5 * new_row.get('lag_1', pred) +
                0.25 * new_row.get('lag_2', pred) +
                0.125 * new_row.get('lag_3', pred) +
                0.0625 * new_row.get('lag_4', pred)
        )

        new_row['volume'] = pred

        # 追加
        new_row_df = pd.DataFrame([new_row], index=[next_date])
        for col in feature_cols:
            if col not in new_row_df.columns:
                new_row_df[col] = 0
        current_data = pd.concat([current_data, new_row_df])

    next_dates = [last_data.index[-1] + timedelta(days=i + 1) for i in range(n_days)]
    return pd.Series(predictions, index=next_dates)


# ============================================
# 5. 主程序
# ============================================

# 加载模型和数据
with st.spinner("正在训练模型..."):
    model, feature_cols, df_feat, X_test, y_test, y_pred, mae, mape = train_model()

# 显示评估指标
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE（件）", f"{mae:.0f}")
col2.metric("MAPE", f"{mape:.1f}%")
col3.metric("训练样本", len(df_feat))
col4.metric("特征数量", len(feature_cols))

st.markdown("---")

# 预测对比图
st.subheader("📈 模型评估：预测 vs 实际（测试集）")

fig1, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(y_test.index, y_test.values, label='实际货量', linewidth=1)
ax1.plot(y_test.index, y_pred, label='预测货量', linewidth=1, alpha=0.8)
ax1.set_xlabel('日期')
ax1.set_ylabel('货量（件）')
ax1.set_title('货量预测结果对比（测试集）')
ax1.legend()
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# 误差分布
if show_details:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 误差分布")
        errors = y_test.values - y_pred
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.hist(errors, bins=30, edgecolor='black', color='steelblue')
        ax2.set_xlabel('误差（件）')
        ax2.set_ylabel('频次')
        ax2.set_title('预测误差分布')
        st.pyplot(fig2)

    with col2:
        st.subheader("🎯 误差 vs 预测值")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.scatter(y_pred, errors, alpha=0.5, color='steelblue')
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_xlabel('预测货量（件）')
        ax3.set_ylabel('误差（件）')
        ax3.set_title('误差-预测值散点图')
        st.pyplot(fig3)

# 特征重要性
if show_feature_importance:
    st.subheader("🔑 特征重要性 Top 15")

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    fig4, ax4 = plt.subplots(figsize=(10, 8))
    top_features = importance_df.head(15)
    ax4.barh(top_features['feature'], top_features['importance'], color='steelblue')
    ax4.invert_yaxis()
    ax4.set_xlabel('重要性分数')
    ax4.set_title('特征重要性 Top 15')
    st.pyplot(fig4)

# 未来预测
st.markdown("---")
st.subheader(f"🔮 未来 {n_days} 天货量预测")

if st.button("开始预测", type="primary"):
    with st.spinner("正在滚动预测..."):
        future_pred = rolling_forecast(model, df_feat, feature_cols, n_days=n_days)

    # 显示预测结果表格
    result_df = pd.DataFrame({
        '日期': future_pred.index.strftime('%Y-%m-%d'),
        '预测货量（件）': future_pred.values.astype(int)
    })
    st.dataframe(result_df, use_container_width=True)

    # 可视化
    fig5, ax5 = plt.subplots(figsize=(12, 5))
    ax5.plot(future_pred.index, future_pred.values, marker='o', linewidth=2, color='steelblue')
    ax5.set_xlabel('日期')
    ax5.set_ylabel('预测货量（件）')
    ax5.set_title(f'未来 {n_days} 天货量预测')
    ax5.grid(True, alpha=0.3)

    # 添加数值标签
    for i, (date, val) in enumerate(future_pred.items()):
        ax5.annotate(f'{int(val)}', (date, val), textcoords="offset points", xytext=(0, 10), ha='center')

    st.pyplot(fig5)

    # 显示预测明细
    with st.expander("查看预测明细"):
        st.write("### 滚动预测过程说明")
        st.markdown("""
        - **滚动预测策略**：每次只预测1天，用预测值作为输入预测下一天
        - **为什么要滚动**：模拟真实业务逐日决策，符合滞后特征依赖
        - **误差说明**：滚动预测的误差会随步数累积，这是正常现象
        """)

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>📦 物流货量预测Demo | 基于XGBoost + 滚动预测 | 代码开源 | 作者：肖元（数量经济学博士）</p>
</div>
""", unsafe_allow_html=True)