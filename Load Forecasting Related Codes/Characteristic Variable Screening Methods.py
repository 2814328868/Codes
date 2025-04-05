import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# 数据加载与预处理
data = pd.read_csv('预处理后数据.csv')  # 替换为你的数据路径

# 提取电负荷、热负荷、冷负荷及气候数据
load_values = data['电负荷'].values.reshape(-1, 1)
heat_values = data['热负荷'].values.reshape(-1, 1)
cool_values = data['冷负荷'].values.reshape(-1, 1)
climate_features = data[['Pressure', 'Temperature', 'Relative Humidity', 'Wind Direction',
                         'Precipitable Water', '云的类型', '可再生能源', 'Hour', 'Wind Speed',
                         'Clearsky DNI', '是否节假日', '季节']]

# 初始化OneHotEncoder并对'是否节假日'和'季节'进行独热编码
onehot_encoder = OneHotEncoder(sparse_output=False)
climate_features[['是否节假日', '季节']] = climate_features[['是否节假日', '季节']].astype(str)
encoded_features = onehot_encoder.fit_transform(climate_features[['是否节假日', '季节']])
encoded_df = pd.DataFrame(encoded_features, columns=onehot_encoder.get_feature_names_out(['是否节假日', '季节']))

# 删除原始'是否节假日'和'季节'列并添加独热编码后的数据
climate_features = pd.concat([climate_features.drop(columns=['是否节假日', '季节']),
                              encoded_df], axis=1)

# 定义互信息筛选函数
def mutual_info_selection(X, y, num_features):
    """计算互信息得分并返回最重要的特征"""
    mi = mutual_info_regression(X, y, discrete_features='auto')
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    selected_features = mi_series.head(num_features).index.tolist()
    return selected_features, mi_series

# 定义LightGBM特征重要性筛选函数
def lightgbm_feature_importance(X, y, num_features):
    """训练LightGBM并返回最重要的特征"""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1
    }
    dataset = lgb.Dataset(X, label=y)

    # 使用callbacks实现early stopping
    callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=False)]
    cv_results = lgb.cv(params, dataset, num_boost_round=100, nfold=5, stratified=False,
                        callbacks=callbacks, return_cvbooster=True)

    # 检查最佳轮数
    if 'valid rmse-mean' in cv_results:
        best_num_boost_round = len(cv_results['valid rmse-mean'])
    else:
        best_num_boost_round = 100  # 默认值

    # 训练模型
    model = lgb.train(params, dataset, num_boost_round=best_num_boost_round)
    feature_importances = model.feature_importance()
    fi_series = pd.Series(feature_importances, index=X.columns).sort_values(ascending=False)
    selected_features = fi_series.head(num_features).index.tolist()

    return selected_features, fi_series



# 定义最终特征筛选函数
def final_feature_selection(X, y, mi_top_k, lgb_top_k, output_prefix):
    """结合互信息与LightGBM筛选特征，并保存特征得分"""
    # 1. 互信息快速筛选
    mi_selected, mi_scores = mutual_info_selection(X, y, mi_top_k)
    mi_scores.to_csv(f"{output_prefix}_mutual_info_scores.csv", encoding='utf-8-sig')
    print(f"\n互信息筛选得分已保存到 {output_prefix}_mutual_info_scores.csv")

    X_reduced = X[mi_selected]

    # 2. LightGBM进一步筛选
    lgb_selected, lgb_scores = lightgbm_feature_importance(X_reduced, y, lgb_top_k)
    lgb_scores.to_csv(f"{output_prefix}_lightgbm_scores.csv", encoding='utf-8-sig')
    print(f"\nLightGBM筛选得分已保存到 {output_prefix}_lightgbm_scores.csv")

    # 返回最终筛选的特征及其重要性
    return lgb_selected, lgb_scores

# 筛选不同负荷的特征
mi_top_k = 10  # 互信息筛选出的特征数量
lgb_top_k = 5  # LightGBM最终筛选出的特征数量

# 电负荷
electric_selected_features, electric_scores = final_feature_selection(
    climate_features, load_values.ravel(), mi_top_k, lgb_top_k, "electric")
print("\n电负荷最终选择的特征：", electric_selected_features)

# 热负荷
heat_selected_features, heat_scores = final_feature_selection(
    climate_features, heat_values.ravel(), mi_top_k, lgb_top_k, "heat")
print("\n热负荷最终选择的特征：", heat_selected_features)

# 冷负荷
cold_selected_features, cold_scores = final_feature_selection(
    climate_features, cool_values.ravel(), mi_top_k, lgb_top_k, "cold")
print("\n冷负荷最终选择的特征：", cold_selected_features)

