import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Bidirectional, Concatenate, MultiHeadAttention, LayerNormalization, Flatten,Layer
from tensorflow.keras.regularizers import l2
from skopt import gp_minimize
from skopt.space import Real, Integer

# 数据加载与预处理
data = pd.read_csv('预处理后数据.csv')  # 替换为你的数据路径

from sklearn.preprocessing import MinMaxScaler

# 提取电负荷、热负荷和冷负荷及气候数据
load_values = data['电负荷'].values.reshape(-1, 1)
heat_values = data['热负荷'].values.reshape(-1, 1)
cool_values = data['冷负荷'].values.reshape(-1, 1)

# 分别定义电负荷、热负荷和冷负荷的气候特征
climate_features_electric = data[['Temperature', 'Precipitable Water', 'Wind Direction', 'Relative Humidity', '可再生能源']].values
climate_features_heat = data[['可再生能源', 'Temperature', 'Wind Direction',  'Relative Humidity','Pressure', 'Precipitable Water']].values
climate_features_cool = data[['Temperature', 'Precipitable Water', 'Relative Humidity', 'Wind Direction', '可再生能源', 'Hour']].values

# 数据归一化
scaler_load = MinMaxScaler()
load_scaled = scaler_load.fit_transform(load_values)
heat_scaled = scaler_load.fit_transform(heat_values)
cool_scaled = scaler_load.fit_transform(cool_values)

scaler_climate_electric = MinMaxScaler()
climate_scaled_electric = scaler_climate_electric.fit_transform(climate_features_electric)

scaler_climate_heat = MinMaxScaler()
climate_scaled_heat = scaler_climate_heat.fit_transform(climate_features_heat)

scaler_climate_cool = MinMaxScaler()
climate_scaled_cool = scaler_climate_cool.fit_transform(climate_features_cool)

# 数据集划分
train_size = int(len(load_scaled) * 0.8)
val_size = int(len(load_scaled) * 0.1)

# 电负荷训练集、验证集和测试集
train_data_electric = load_scaled[:train_size]
val_data_electric = load_scaled[train_size:train_size + val_size]
test_data_electric = load_scaled[train_size + val_size:]

# 热负荷训练集、验证集和测试集
train_data_heat = heat_scaled[:train_size]
val_data_heat = heat_scaled[train_size:train_size + val_size]
test_data_heat = heat_scaled[train_size + val_size:]

# 冷负荷训练集、验证集和测试集
train_data_cool = cool_scaled[:train_size]
val_data_cool = cool_scaled[train_size:train_size + val_size]
test_data_cool = cool_scaled[train_size + val_size:]

# 电负荷气候特征训练集、验证集和测试集
train_climate_electric = climate_scaled_electric[:train_size]
val_climate_electric = climate_scaled_electric[train_size:train_size + val_size]
test_climate_electric = climate_scaled_electric[train_size + val_size:]

# 热负荷气候特征训练集、验证集和测试集
train_climate_heat = climate_scaled_heat[:train_size]
val_climate_heat = climate_scaled_heat[train_size:train_size + val_size]
test_climate_heat = climate_scaled_heat[train_size + val_size:]

# 冷负荷气候特征训练集、验证集和测试集
train_climate_cool = climate_scaled_cool[:train_size]
val_climate_cool = climate_scaled_cool[train_size:train_size + val_size]
test_climate_cool = climate_scaled_cool[train_size + val_size:]

# 修改后的量子损失函数，带拐点惩罚      penalty_factor越大, threshold=越小，惩罚越强
def average_quantile_loss(q, y_true, y_pred, lambda_reg=0.01, weights=None, penalty_factor=8, threshold=0.01):
    e = y_true - y_pred
    loss = tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))

    if weights is not None:
        loss *= weights

    large_error_mask = tf.abs(e) > threshold
    penalized_loss = penalty_factor * tf.reduce_mean(tf.where(large_error_mask, tf.square(e), tf.zeros_like(e)))
    loss += penalized_loss

    if weights is not None:
        l2_loss = lambda_reg * tf.reduce_sum(tf.square(weights))
        loss += l2_loss

    return loss


# 动态注意力机制类定义
class TimeStepAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(TimeStepAttention, self).__init__()
        self.units = units
        self.W = Dense(units)
        self.V = Dense(1)

    def call(self, inputs):
        score = self.V(tf.nn.tanh(self.W(inputs)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# 定义输入数据的形状，使用 None 来表示时间步长的动态性
input_shape = (None,  1)  # None 表示时间步长是动态的，可以接受不同长度的序列
# 为电负荷、热负荷、冷负荷定义输入
input_electric = Input(shape=input_shape)  # 电负荷输入
input_heat = Input(shape=input_shape)  # 热负荷输入
input_cool = Input(shape=input_shape)  # 冷负荷输入

# 假设气候特征数据是多维的，形状为 (samples, features)，通常是 (samples, 6) 之类的
climate_shape_electric = (5,)
climate_shape_cool = (6,)
climate_shape_heat = (6,)

# 构建多任务模型
def build_mtl_cnn_bilstm_with_timestep_attention(input_electric, input_cold, input_heat, climate_shape_electric, climate_shape_cool, climate_shape_heat, dropout_rate, l2_reg):
    input_climate_electric = Input(shape=climate_shape_electric)  # 使用传入的电负荷气候特征形状
    input_climate_cool = Input(shape=climate_shape_cool)          # 使用传入的冷负荷气候特征形状
    input_climate_heat = Input(shape=climate_shape_heat)          # 使用传入的热负荷气候特征形状

    # 电负荷的卷积和双向LSTM处理
    x_electric = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(
        input_electric)
    x_electric = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg)))(
        x_electric)  # 保证 return_sequences=True
    x_electric = Dropout(dropout_rate)(x_electric)
    x_electric = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg)))(
        x_electric)  # 保证 return_sequences=True
    x_electric = Dropout(dropout_rate)(x_electric)

    # 冷负荷的卷积和双向LSTM处理
    x_cold = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(input_cold)
    x_cold = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg)))(x_cold)  # 保证 return_sequences=True
    x_cold = Dropout(dropout_rate)(x_cold)
    x_cold = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg)))(x_cold)  # 保证 return_sequences=True
    x_cold = Dropout(dropout_rate)(x_cold)

    # 热负荷的卷积和双向LSTM处理
    x_heat = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(input_heat)
    x_heat = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg)))(x_heat)  # 保证 return_sequences=True
    x_heat = Dropout(dropout_rate)(x_heat)
    x_heat = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg)))(x_heat)  # 保证 return_sequences=True
    x_heat = Dropout(dropout_rate)(x_heat)

    # 修改 TimeStepAttention 层的调用，确保返回两个输出（context_vector 和 attention_weights）
    attention_electric, attention_weights_electric = TimeStepAttention(64)(x_electric)
    attention_cold, attention_weights_cold = TimeStepAttention(64)(x_cold)
    attention_heat, attention_weights_heat = TimeStepAttention(64)(x_heat)

    # 对每个负荷的注意力输出应用 LayerNormalization
    attention_output_electric = LayerNormalization()(attention_electric)  # 规范化电负荷的注意力输出
    attention_output_cold = LayerNormalization()(attention_cold)  # 规范化冷负荷的注意力输出
    attention_output_heat = LayerNormalization()(attention_heat)  # 规范化热负荷的注意力输出

    # 各负荷结合各自气候特征
    combined_cool = Concatenate()([attention_output_cold, input_climate_cool])
    combined_heat = Concatenate()([attention_output_heat, input_climate_heat])
    combined_electric = Concatenate()([attention_output_electric, input_climate_electric])

    # 对每种负荷的组合输出应用 Dropout
    combined_cool = Dropout(dropout_rate)(combined_cool)
    combined_heat = Dropout(dropout_rate)(combined_heat)
    combined_electric = Dropout(dropout_rate)(combined_electric)

    # 重新定义模型的输出
    output_electric_5 = Dense(1, name='electric_q_5', kernel_regularizer=l2(l2_reg))(combined_electric)
    output_electric_95 = Dense(1, name='electric_q_95', kernel_regularizer=l2(l2_reg))(combined_electric)
    output_electric_2_5 = Dense(1, name='electric_q_2_5', kernel_regularizer=l2(l2_reg))(combined_electric)
    output_electric_97_5 = Dense(1, name='electric_q_97_5', kernel_regularizer=l2(l2_reg))(combined_electric)

    output_heat_5 = Dense(1, name='heat_q_5', kernel_regularizer=l2(l2_reg))(combined_heat)
    output_heat_95 = Dense(1, name='heat_q_95', kernel_regularizer=l2(l2_reg))(combined_heat)
    output_heat_2_5 = Dense(1, name='heat_q_2_5', kernel_regularizer=l2(l2_reg))(combined_heat)
    output_heat_97_5 = Dense(1, name='heat_q_97_5', kernel_regularizer=l2(l2_reg))(combined_heat)

    output_cool_5 = Dense(1, name='cool_q_5', kernel_regularizer=l2(l2_reg))(combined_cool)
    output_cool_95 = Dense(1, name='cool_q_95', kernel_regularizer=l2(l2_reg))(combined_cool)
    output_cool_2_5 = Dense(1, name='cool_q_2_5', kernel_regularizer=l2(l2_reg))(combined_cool)
    output_cool_97_5 = Dense(1, name='cool_q_97_5', kernel_regularizer=l2(l2_reg))(combined_cool)

    model = Model(
        inputs=[input_electric, input_cold, input_heat, input_climate_electric, input_climate_cool, input_climate_heat],
        outputs=[
            output_electric_5, output_electric_95, output_electric_2_5, output_electric_97_5,
            output_heat_5, output_heat_95, output_heat_2_5, output_heat_97_5,
            output_cool_5, output_cool_95, output_cool_2_5, output_cool_97_5
        ]
    )

    return model


# 定义贝叶斯优化的超参数空间，包括L2正则化
param_space = [
    Integer(16, 128, name='batch_size'),
    Integer(10, 150, name='epochs'),
    Real(1e-5, 1e-2, prior='log-uniform', name='learning_rate'),
    Real(0.1, 0.5, name='dropout_rate'),
    Real(1e-6, 1e-2, prior='log-uniform', name='l2_reg')
]

# 添加计算AIS的函数
def calculate_ais(y_true, lower_bounds, upper_bounds, alpha):
    """
    计算Average Interval Score (AIS).

    参数:
        y_true (np.array): 真实值数组.
        lower_bounds (np.array): 预测区间的下界数组.
        upper_bounds (np.array): 预测区间的上界数组.
        alpha (float): 显著性水平（通常取0.05或0.1）.

    返回:
        float: AIS得分.
    """
    # 样本数量
    n = len(y_true)

    # 计算AIS
    ais = np.mean((upper_bounds - lower_bounds) +
                  (2 / alpha) * (lower_bounds - y_true) * (y_true < lower_bounds) +
                  (2 / alpha) * (y_true - upper_bounds) * (y_true > upper_bounds))

    return ais

# 全局变量初始化
iteration = 0  # 初始化轮次计数器

# 电负荷目标函数
def objective_load(params):
    batch_size, epochs, learning_rate, dropout_rate, l2_reg = params
    global iteration
    iteration += 1  # 更新轮次计数
    # 构建模型
    model = build_mtl_cnn_bilstm_with_timestep_attention(input_electric, input_cool, input_heat, climate_shape_electric, climate_shape_cool, climate_shape_heat, dropout_rate, l2_reg)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss={
            'electric_q_5': lambda y_true, y_pred: average_quantile_loss(0.05, y_true, y_pred),
            'electric_q_95': lambda y_true, y_pred: average_quantile_loss(0.95, y_true, y_pred),
            'electric_q_2_5': lambda y_true, y_pred: average_quantile_loss(0.025, y_true, y_pred),
            'electric_q_97_5': lambda y_true, y_pred: average_quantile_loss(0.975, y_true, y_pred),
            'cool_q_5': lambda y_true, y_pred: average_quantile_loss(0.05, y_true, y_pred),
            'cool_q_95': lambda y_true, y_pred: average_quantile_loss(0.95, y_true, y_pred),
            'cool_q_2_5': lambda y_true, y_pred: average_quantile_loss(0.025, y_true, y_pred),
            'cool_q_97_5': lambda y_true, y_pred: average_quantile_loss(0.975, y_true, y_pred),
            'heat_q_5': lambda y_true, y_pred: average_quantile_loss(0.05, y_true, y_pred),
            'heat_q_95': lambda y_true, y_pred: average_quantile_loss(0.95, y_true, y_pred),
            'heat_q_2_5': lambda y_true, y_pred: average_quantile_loss(0.025, y_true, y_pred),
            'heat_q_97_5': lambda y_true, y_pred: average_quantile_loss(0.975, y_true, y_pred)
        }
    )

    # 训练模型
    model.fit(
        [
            train_data_electric, train_data_cool, train_data_heat,
            train_climate_electric, train_climate_cool, train_climate_heat
        ],
        [
            train_data_electric, train_data_electric, train_data_electric, train_data_electric,  # 电负荷四个分位数
            train_data_heat, train_data_heat, train_data_heat, train_data_heat,  # 热负荷四个分位数
            train_data_cool, train_data_cool, train_data_cool, train_data_cool  # 冷负荷四个分位数
        ],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(
            [
                val_data_electric, val_data_cool, val_data_heat,
                val_climate_electric, val_climate_cool, val_climate_heat
            ],
            [
                val_data_electric, val_data_electric, val_data_electric, val_data_electric,  # 电负荷四个分位数
                val_data_heat, val_data_heat, val_data_heat, val_data_heat,  # 热负荷四个分位数
                val_data_cool, val_data_cool, val_data_cool, val_data_cool  # 冷负荷四个分位数
            ]
        ),
        verbose=0
    )

    # 预测和评估
    predictions = model.predict([val_data_electric, val_data_cool, val_data_heat, val_climate_electric, val_climate_cool, val_climate_heat])
    predicted_electric_5 = scaler_load.inverse_transform(predictions[0])  # 5%
    predicted_electric_95 = scaler_load.inverse_transform(predictions[1])  # 95%
    predicted_electric_2_5 = scaler_load.inverse_transform(predictions[2])  # 2.5%
    predicted_electric_97_5 = scaler_load.inverse_transform(predictions[3])  # 97.5%
    true_electric = scaler_load.inverse_transform(val_data_electric.reshape(-1, 1))  # 提取每个样本最后一个时间步的电负荷值

    # 计算评价指标
    PICP_load = np.mean((true_electric >= predicted_electric_5) & (true_electric <= predicted_electric_95))
    PINAW_load = np.mean(predicted_electric_95 - predicted_electric_5) / (np.max(true_electric) - np.min(true_electric) + 1e-8)
    ais_5_95 = calculate_ais(true_electric, predicted_electric_5, predicted_electric_95, alpha=0.1)
    rmse_load = np.sqrt(mean_squared_error(true_electric, predicted_electric_95))
    print(f"第 {iteration} 轮次: PICP={PICP_load:.4f}, AIS={ais_5_95:.4f}, PINAW={PINAW_load:.4f}, RMSE={rmse_load:.4f}" )

    # 优化目标：负的 PICP + MPIW + PINAW + RMSE
    return -PICP_load + 2*ais_5_95 + 2*PINAW_load + 1.5*rmse_load

# 运行贝叶斯优化
results_load = gp_minimize(objective_load, param_space, n_calls=30, random_state=0)
# 打印最佳超参数和结果
print(f"电负荷最佳超参数: {results_load.x}, 最佳结果: {-results_load.fun}")


# # 热负荷目标函数
def objective_heat(params):
    batch_size, epochs, learning_rate, dropout_rate, l2_reg = params
    global iteration
    iteration += 1  # 更新轮次计数
    # 构建模型
    model = build_mtl_cnn_bilstm_with_timestep_attention(input_electric, input_cool, input_heat, climate_shape_electric, climate_shape_cool, climate_shape_heat, dropout_rate, l2_reg)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss={
            'electric_q_5': lambda y_true, y_pred: average_quantile_loss(0.05, y_true, y_pred),
            'electric_q_95': lambda y_true, y_pred: average_quantile_loss(0.95, y_true, y_pred),
            'electric_q_2_5': lambda y_true, y_pred: average_quantile_loss(0.025, y_true, y_pred),
            'electric_q_97_5': lambda y_true, y_pred: average_quantile_loss(0.975, y_true, y_pred),
            'cool_q_5': lambda y_true, y_pred: average_quantile_loss(0.05, y_true, y_pred),
            'cool_q_95': lambda y_true, y_pred: average_quantile_loss(0.95, y_true, y_pred),
            'cool_q_2_5': lambda y_true, y_pred: average_quantile_loss(0.025, y_true, y_pred),
            'cool_q_97_5': lambda y_true, y_pred: average_quantile_loss(0.975, y_true, y_pred),
            'heat_q_5': lambda y_true, y_pred: average_quantile_loss(0.05, y_true, y_pred),
            'heat_q_95': lambda y_true, y_pred: average_quantile_loss(0.95, y_true, y_pred),
            'heat_q_2_5': lambda y_true, y_pred: average_quantile_loss(0.025, y_true, y_pred),
            'heat_q_97_5': lambda y_true, y_pred: average_quantile_loss(0.975, y_true, y_pred)
        }
    )

# 训练模型
    model.fit(
        [
            train_data_electric, train_data_cool, train_data_heat,
            train_climate_electric, train_climate_cool, train_climate_heat
        ],
        [
            train_data_electric, train_data_electric, train_data_electric, train_data_electric,  # 电负荷四个分位数
            train_data_heat, train_data_heat, train_data_heat, train_data_heat,  # 热负荷四个分位数
            train_data_cool, train_data_cool, train_data_cool, train_data_cool  # 冷负荷四个分位数
        ],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(
            [
                val_data_electric, val_data_cool, val_data_heat,
                val_climate_electric, val_climate_cool, val_climate_heat
            ],
            [
                val_data_electric, val_data_electric, val_data_electric, val_data_electric,  # 电负荷四个分位数
                val_data_heat, val_data_heat, val_data_heat, val_data_heat,  # 热负荷四个分位数
                val_data_cool, val_data_cool, val_data_cool, val_data_cool  # 冷负荷四个分位数
            ]
        ),
        verbose=0
    )

    predictions = model.predict([val_data_electric, val_data_cool, val_data_heat, val_climate_electric, val_climate_cool, val_climate_heat])
    predicted_heat_5 = scaler_load.inverse_transform(predictions[4])   # 5%
    predicted_heat_95 = scaler_load.inverse_transform(predictions[5])  # 95%
    predicted_heat_2_5 = scaler_load.inverse_transform(predictions[6]) # 2.5%
    predicted_heat_97_5 = scaler_load.inverse_transform(predictions[7]) # 97.5%
    true_heat = scaler_load.inverse_transform(val_data_heat.reshape(-1, 1))  # 提取每个样本最后一个时间步的电负荷值

    PICP_heat = np.mean((true_heat >= predicted_heat_5) & (true_heat <= predicted_heat_95))
    PINAW_heat = np.mean(predicted_heat_95 - predicted_heat_5) / (np.max(true_heat) - np.min(true_heat)+ 1e-8)
    ais_heat = calculate_ais(true_heat, predicted_heat_5, predicted_heat_95, alpha=0.1)
    rmse_heat = np.sqrt(mean_squared_error(true_heat, predicted_heat_95))
    print(f"第 {iteration} 轮次: PICP={PICP_heat:.4f}, AIS={ais_heat:.4f}, PINAW={PINAW_heat:.4f}, RMSE={rmse_heat:.4f}")

    return -PICP_heat + 2*ais_heat + 2*PINAW_heat + 1.5*rmse_heat
    # 打印当前轮次结果

# 运行贝叶斯优化
results_heat = gp_minimize(objective_heat, param_space, n_calls=30, random_state=0)
# 打印最佳超参数和结果
print(f"热负荷最佳超参数: {results_heat.x}, 最佳结果: {-results_heat.fun}")

# 冷负荷目标函数
def objective_cool(params):
    batch_size, epochs, learning_rate, dropout_rate, l2_reg = params
    global iteration
    iteration += 1  # 更新轮次计数
    # 构建模型
    model = build_mtl_cnn_bilstm_with_timestep_attention(input_electric, input_cool, input_heat, climate_shape_electric, climate_shape_cool, climate_shape_heat, dropout_rate, l2_reg)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss={
            'electric_q_5': lambda y_true, y_pred: average_quantile_loss(0.05, y_true, y_pred),
            'electric_q_95': lambda y_true, y_pred: average_quantile_loss(0.95, y_true, y_pred),
            'electric_q_2_5': lambda y_true, y_pred: average_quantile_loss(0.025, y_true, y_pred),
            'electric_q_97_5': lambda y_true, y_pred: average_quantile_loss(0.975, y_true, y_pred),
            'cool_q_5': lambda y_true, y_pred: average_quantile_loss(0.05, y_true, y_pred),
            'cool_q_95': lambda y_true, y_pred: average_quantile_loss(0.95, y_true, y_pred),
            'cool_q_2_5': lambda y_true, y_pred: average_quantile_loss(0.025, y_true, y_pred),
            'cool_q_97_5': lambda y_true, y_pred: average_quantile_loss(0.975, y_true, y_pred),
            'heat_q_5': lambda y_true, y_pred: average_quantile_loss(0.05, y_true, y_pred),
            'heat_q_95': lambda y_true, y_pred: average_quantile_loss(0.95, y_true, y_pred),
            'heat_q_2_5': lambda y_true, y_pred: average_quantile_loss(0.025, y_true, y_pred),
            'heat_q_97_5': lambda y_true, y_pred: average_quantile_loss(0.975, y_true, y_pred)
        }
    )

    # 训练模型
    model.fit(
        [
            train_data_electric, train_data_cool, train_data_heat,
            train_climate_electric, train_climate_cool, train_climate_heat
        ],
        [
            train_data_electric, train_data_electric, train_data_electric, train_data_electric,  # 电负荷四个分位数
            train_data_heat, train_data_heat, train_data_heat, train_data_heat,  # 热负荷四个分位数
            train_data_cool, train_data_cool, train_data_cool, train_data_cool  # 冷负荷四个分位数
        ],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(
            [
                val_data_electric, val_data_cool, val_data_heat,
                val_climate_electric, val_climate_cool, val_climate_heat
            ],
            [
                val_data_electric, val_data_electric, val_data_electric, val_data_electric,  # 电负荷四个分位数
                val_data_heat, val_data_heat, val_data_heat, val_data_heat,  # 热负荷四个分位数
                val_data_cool, val_data_cool, val_data_cool, val_data_cool  # 冷负荷四个分位数
            ]
        ),
        verbose=0
    )

    predictions = model.predict([val_data_electric, val_data_cool, val_data_heat, val_climate_electric, val_climate_cool, val_climate_heat])
    predicted_cool_5 = scaler_load.inverse_transform(predictions[8])   # 5%
    predicted_cool_95 = scaler_load.inverse_transform(predictions[9])  # 95%
    predicted_cool_2_5 = scaler_load.inverse_transform(predictions[10]) # 2.5%
    predicted_cool_97_5 = scaler_load.inverse_transform(predictions[11]) # 97.5%
    true_cool = scaler_load.inverse_transform(val_data_cool.reshape(-1, 1))  # 提取每个样本最后一个时间步的电负荷值

    PICP_cool = np.mean((true_cool >= predicted_cool_5) & (true_cool <= predicted_cool_95))
    PINAW_cool = np.mean(predicted_cool_95 - predicted_cool_5) / (np.max(true_cool) - np.min(true_cool)+ 1e-8)
    ais_cool = calculate_ais(true_cool, predicted_cool_5, predicted_cool_95, alpha=0.1)
    rmse_cool = np.sqrt(mean_squared_error(true_cool, predicted_cool_95))
    print(f"第 {iteration} 轮次: PICP={PICP_cool:.4f}, AIS={ais_cool:.4f}, PINAW={PINAW_cool:.4f}, RMSE={rmse_cool:.4f}")

    return -PICP_cool + 2*ais_cool + 2*PINAW_cool + 1.5*rmse_cool

# 运行贝叶斯优化
results_cool = gp_minimize(objective_cool, param_space, n_calls=30, random_state=0)
# 打印最佳超参数和结果
print(f"冷负荷最佳超参数: {results_cool.x}, 最佳结果: {-results_cool.fun}")