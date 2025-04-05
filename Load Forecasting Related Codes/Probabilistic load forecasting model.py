import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense,  Dropout, Bidirectional, Concatenate, LayerNormalization
from tensorflow.keras.regularizers import l2
import csv

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
scaler_heat = MinMaxScaler()
heat_scaled = scaler_heat.fit_transform(heat_values)
scaler_cool = MinMaxScaler()
cool_scaled = scaler_cool.fit_transform(cool_values)

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


# 修改后的量子损失函数，带拐点惩罚      penalty_factor越大, threshold=越小，惩罚越强     8-0.001已知挺好
def average_quantile_loss(q, y_true, y_pred, lambda_reg=0.01, weights=None, penalty_factor=0, threshold=0):
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
def build_mtl_cnn_bilstm_with_timestep_attention(input_electric, input_cool, input_heat, climate_shape_electric, climate_shape_cool, climate_shape_heat, dropout_rate, l2_reg):
    input_climate_electric = Input(shape=climate_shape_electric)  # 使用传入的电负荷气候特征形状
    input_climate_cool = Input(shape=climate_shape_cool)          # 使用传入的冷负荷气候特征形状
    input_climate_heat = Input(shape=climate_shape_heat)          # 使用传入的热负荷气候特征形状

    # 电负荷的卷积和双向LSTM处理
    x_electric = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal', kernel_regularizer=l2(l2_reg))(
        input_electric)
    x_electric = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg)))(
        x_electric)  # 保证 return_sequences=True
    x_electric = Dropout(dropout_rate)(x_electric)
    x_electric = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg)))(
        x_electric)  # 保证 return_sequences=True
    x_electric = Dropout(dropout_rate)(x_electric)

    # 冷负荷的卷积和双向LSTM处理
    x_cool = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal', kernel_regularizer=l2(l2_reg))(
        input_cool)
    x_cool = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg)))(
        x_cool)  # 保证 return_sequences=True
    x_cold = Dropout(dropout_rate)(x_cool)
    x_cold = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg)))(
        x_cold)  # 保证 return_sequences=True
    x_cold = Dropout(dropout_rate)(x_cold)

    # 热负荷的卷积和双向LSTM处理
    x_heat = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal', kernel_regularizer=l2(l2_reg))(
        input_heat)
    x_heat = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg)))(
        x_heat)  # 保证 return_sequences=True
    x_heat = Dropout(dropout_rate)(x_heat)
    x_heat = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_reg)))(
        x_heat)  # 保证 return_sequences=True
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
        inputs=[input_electric, input_cool, input_heat, input_climate_electric, input_climate_cool, input_climate_heat],
        outputs=[
            output_electric_5, output_electric_95, output_electric_2_5, output_electric_97_5,
            output_heat_5, output_heat_95, output_heat_2_5, output_heat_97_5,
            output_cool_5, output_cool_95, output_cool_2_5, output_cool_97_5
        ]
    )

    # 编译模型   # 使用 `create_quantile_loss_functions` 动态生成损失函数字典
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss={
            'electric_q_5': lambda y_true, y_pred: average_quantile_loss(0.05, y_true, y_pred),
            'electric_q_95': lambda y_true, y_pred: average_quantile_loss(0.95, y_true, y_pred),
            'electric_q_2_5': lambda y_true, y_pred: average_quantile_loss(0.025, y_true, y_pred),
            'electric_q_97_5': lambda y_true, y_pred: average_quantile_loss(0.975, y_true, y_pred),
            'heat_q_5': lambda y_true, y_pred: average_quantile_loss(0.05, y_true, y_pred),
            'heat_q_95': lambda y_true, y_pred: average_quantile_loss(0.95, y_true, y_pred),
            'heat_q_2_5': lambda y_true, y_pred: average_quantile_loss(0.025, y_true, y_pred),
            'heat_q_97_5': lambda y_true, y_pred: average_quantile_loss(0.975, y_true, y_pred),
            'cool_q_5': lambda y_true, y_pred: average_quantile_loss(0.05, y_true, y_pred),
            'cool_q_95': lambda y_true, y_pred: average_quantile_loss(0.95, y_true, y_pred),
            'cool_q_2_5': lambda y_true, y_pred: average_quantile_loss(0.025, y_true, y_pred),
            'cool_q_97_5': lambda y_true, y_pred: average_quantile_loss(0.975, y_true, y_pred),
        }
    )
    return model


# 定义各个负荷的超参数
params_dict = {
    '电负荷': {
        'batch_size': 71,
        'epochs': 150,
        'learning_rate': 0.0006556,
        'dropout_rate': 0.1,
        'l2_reg': 1.0144e-06
    },
    '热负荷': {
        'batch_size': 128,
        'epochs': 137,
        'learning_rate': 0.002578,
        'dropout_rate': 0.1,
        'l2_reg': 1e-06
    },
    '冷负荷': {
        'batch_size': 16,
        'epochs': 127,
        'learning_rate': 3.64e-05,
        'dropout_rate': 0.1,
        'l2_reg': 0.000137
    }
}

# loss记录
class LossHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, csv_filename):
        super(LossHistoryCallback, self).__init__()
        self.csv_filename = csv_filename
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        # 保存每个 epoch 的训练和验证 loss
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}:")
        print(f" - Train Loss: {logs.get('loss'):.4f}")
        # 每个 epoch 结束时将 loss 写入 CSV 文件
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if epoch == 0:  # 添加标题行
                writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
            writer.writerow([epoch + 1, train_loss, val_loss])

# 定义函数保存预测结果到 CSV 文件
def save_predictions_to_csv(predictions, true_values, file_name):
    data = {
        'True Value': true_values.flatten(),
        'Predicted q_5': predictions[0].flatten(),
        'Predicted q_95': predictions[1].flatten(),
        'Predicted q_2_5': predictions[2].flatten(),
        'Predicted q_97_5': predictions[3].flatten()
    }
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)

def calculate_ais(y_true, lower_bounds, upper_bounds, alpha, scale_min=None, scale_max=None):
    """
    计算归一化的Average Interval Score (AIS).

    参数:
        y_true (np.array): 真实值数组.
        lower_bounds (np.array): 预测区间的下界数组.
        upper_bounds (np.array): 预测区间的上界数组.
        alpha (float): 显著性水平（通常取0.05或0.1）.
        scale_min (float): 数据归一化的最小值.
        scale_max (float): 数据归一化的最大值.

    返回:
        float: AIS得分.
    """
    if scale_min is not None and scale_max is not None:
        # 归一化处理
        y_true = (y_true - scale_min) / (scale_max - scale_min)
        lower_bounds = (lower_bounds - scale_min) / (scale_max - scale_min)
        upper_bounds = (upper_bounds - scale_min) / (scale_max - scale_min)

    # 样本数量
    n = len(y_true)

    # 计算AIS
    ais = np.mean((upper_bounds - lower_bounds) +
                  (2 / alpha) * (lower_bounds - y_true) * (y_true < lower_bounds) +
                  (2 / alpha) * (y_true - upper_bounds) * (y_true > upper_bounds))

    return ais

# 结果存储
results = {}
# 创建和训练包含多头注意力机制的量子预测模型
for key, train_set, val_set, test_set, train_climate, val_climate, test_climate in tqdm(zip(
    ['电负荷', '热负荷', '冷负荷'],
    [train_data_electric, train_data_heat, train_data_cool],
    [val_data_electric, val_data_heat, val_data_cool],
    [test_data_electric, test_data_heat, test_data_cool],
    [train_climate_electric, train_climate_heat, train_climate_cool],
    [val_climate_electric, val_climate_heat, val_climate_cool],
    [test_climate_electric, test_climate_heat, test_climate_cool]
), total=3, desc="训练进度"):

    # 获取当前负荷的超参数
    batch_size = params_dict[key]['batch_size']
    epochs = params_dict[key]['epochs']
    learning_rate = params_dict[key]['learning_rate']
    dropout_rate = params_dict[key]['dropout_rate']
    l2_reg = params_dict[key]['l2_reg']

    # 构建模型
    model = build_mtl_cnn_bilstm_with_timestep_attention(input_electric, input_cool, input_heat, climate_shape_electric, climate_shape_cool, climate_shape_heat, dropout_rate, l2_reg)

    # 设置 CSV 文件名称
    loss_csv_file = f"{key}_loss_history.csv"
    prediction_csv_file = f"{key}_predictions.csv"

    # 初始化 loss 记录回调
    loss_history_callback = LossHistoryCallback(csv_filename=loss_csv_file)
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
        verbose=0,
        callbacks = [loss_history_callback]  # 添加回调
    )

    # 定义函数保存预测结果到 CSV 文件
    def save_predictions_to_csv(predictions, true_values, file_name):
        # 创建一个 DataFrame 来保存真实值和预测值
        data = {
            'True Value': true_values.flatten(),  # 扁平化数组，以确保它是列向量
            'Predicted q_5': predictions[0].flatten(),
            'Predicted q_95': predictions[1].flatten(),
            'Predicted q_2_5': predictions[2].flatten(),
            'Predicted q_97_5': predictions[3].flatten()
        }

        df = pd.DataFrame(data)

        # 保存到 CSV 文件
        df.to_csv(file_name, index=False)

    # 预测
    predictions = model.predict(
        [test_data_electric, test_data_cool, test_data_heat, test_climate_electric, test_climate_cool,test_climate_heat])
    # 为不同负荷提取对应的分位数预测结果
    if key == '电负荷':
        predicted_load_5 = predictions[0]
        predicted_load_95 = predictions[1]
        predicted_load_2_5 = predictions[2]
        predicted_load_97_5 = predictions[3]
    elif key == '热负荷':
        predicted_load_5 = predictions[4]
        predicted_load_95 = predictions[5]
        predicted_load_2_5 = predictions[6]
        predicted_load_97_5 = predictions[7]
    elif key == '冷负荷':
        predicted_load_5 = predictions[8]
        predicted_load_95 = predictions[9]
        predicted_load_2_5 = predictions[10]
        predicted_load_97_5 = predictions[11]

    # 对预测结果进行反归一化
    predicted_load_5 = scaler_load.inverse_transform(predicted_load_5)
    predicted_load_95 = scaler_load.inverse_transform(predicted_load_95)
    predicted_load_2_5 = scaler_load.inverse_transform(predicted_load_2_5)
    predicted_load_97_5 = scaler_load.inverse_transform(predicted_load_97_5)
    true_load = scaler_load.inverse_transform(test_set.reshape(-1, 1))

    # 保存预测结果到 CSV 文件
    save_predictions_to_csv(
        [predicted_load_5, predicted_load_95, predicted_load_2_5, predicted_load_97_5],
        true_load,
        prediction_csv_file
    )
    print(f"{key} 的预测值和loss值已经保存到文件 {loss_csv_file} 和 {prediction_csv_file}。")

    # 计算指标（q_5 与 q_95）
    PICP_5_95 = np.mean((true_load >= predicted_load_5) & (true_load <= predicted_load_95))
    MPIW_5_95 = np.mean(predicted_load_95 - predicted_load_5)
    PINAW_5_95 = MPIW_5_95 / (np.max(true_load) - np.min(true_load))
    ais_score_5_95 = calculate_ais(true_load, predicted_load_5, predicted_load_95, alpha=0.1)

    # 计算指标（q_2_5 与 q_97_5）
    PICP_2_97 = np.mean((true_load >= predicted_load_2_5) & (true_load <= predicted_load_97_5))
    MPIW_2_97 = np.mean(predicted_load_97_5 - predicted_load_2_5)
    PINAW_2_97 = MPIW_2_97 / (np.max(true_load) - np.min(true_load))
    ais_score_2_97 = calculate_ais(true_load, predicted_load_2_5, predicted_load_97_5, alpha=0.05)

    results[key] = {
        'predicted_load_5': predicted_load_5,
        'predicted_load_95': predicted_load_95,
        'predicted_load_2_5': predicted_load_2_5,
        'predicted_load_97_5': predicted_load_97_5,
        'true_load': true_load,'PICP_5_95': PICP_5_95,
        'MPIW_5_95': MPIW_5_95,
        'PINAW_5_95': PINAW_5_95,
        'AIS_5_95': ais_score_5_95,

        'PICP_2_97': PICP_2_97,
        'MPIW_2_97': MPIW_2_97,
        'PINAW_2_97': PINAW_2_97,
        'AIS_2_97': ais_score_2_97
    }

    # 打印结果
    print(f'{key} 结果:')
    print(f'PICP (5% - 95%): {PICP_5_95}')
    print(f'MPIW (5% - 95%): {MPIW_5_95}')
# print(f'Interval Score (5% - 95%): {interval_score_5_95}')
    print(f'PINAW (5% - 95%): {PINAW_5_95}')
    print(f'AIS (5% - 95%): {ais_score_5_95}')
    print(f'PICP (2.5% - 97.5%): {PICP_2_97}')
    print(f'MPIW (2.5% - 97.5%): {MPIW_2_97}')
# print(f'Interval Score (2.5% - 97.5%): {interval_score_2_97}')
    print(f'PINAW (2.5% - 97.5%): {PINAW_2_97}')
    print(f'AIS (2.5% - 97.5%): {ais_score_2_97}\n')

    # 读取 CSV 文件
    loss_data = pd.read_csv(loss_csv_file)

