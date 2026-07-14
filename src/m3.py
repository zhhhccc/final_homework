import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

# 设置随机种子，保证结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def operating_m3():
    set_seed(42)

    df = pd.read_pickle("D:/tasks/final/data_clean.pkl")[['tpep_pickup_datetime']].copy()

    # ==================== 特征工程 ====================
    """
    特征选择说明：
    1. 使用"小时"作为唯一输入特征的原因：
       - 出租车出行需求具有明显的日内周期性（如早晚高峰）
       - 小时级别的粒度既能捕捉出行规律，又不会使特征维度过于复杂
       - 对于"某时段出行需求量"预测任务，小时是最直接且有效的时段划分单位
       - 避免引入过多特征（如天气、节假日等）导致模型过拟合，同时保持计算效率
       - 小时特征作为连续型数值，能够反映时间流动带来的需求变化趋势
    """

    # 提取小时特征
    df['hour'] = df['tpep_pickup_datetime'].dt.hour

    # 统计每个小时的出行需求量（计费次数）
    # 按小时分组统计订单数量，得到每个小时的需求量
    hourly_demand = df.groupby('hour').size().reset_index(name='demand_count')

    # 特征：小时（0-23）
    X = hourly_demand['hour'].values.reshape(-1, 1).astype(np.float32)
    # 目标：该小时的出行需求量
    y = hourly_demand['demand_count'].values.astype(np.float32)

    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"小时分布: {sorted(hourly_demand['hour'].unique())}")
    print(f"需求量范围: {y.min()} - {y.max()}")

    # ==================== 划分训练集和测试集 (8:2) ====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

    # ==================== PyTorch 神经网络 ====================
    class DemandPredictor(nn.Module):
        """简单的三层神经网络用于需求量预测"""
        def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
            super(DemandPredictor, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim)
            )

        def forward(self, x):
            return self.net(x)

    # 转换为PyTorch张量
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).reshape(-1, 1)

    # 创建DataLoader
    batch_size = 16
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = DemandPredictor(input_dim=1, hidden_dim=64, output_dim=1)
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    num_epochs = 500
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # ==================== 绘制训练Loss曲线 ====================
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Neural Network Training Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('D:/tasks/final/outputs/m3_neural_network_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("训练Loss曲线已保存至 outputs/m3_neural_network_loss.png")

    # ==================== 神经网络预测与评估 ====================
    model.eval()
    with torch.no_grad():
        y_pred_nn = model(X_test_t).numpy().flatten()

    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))

    print(f"神经网络 - MAE: {mae_nn:.4f}, RMSE: {rmse_nn:.4f}")

    # ==================== 随机森林回归 ====================
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1  # 使用所有CPU核心加速
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    print(f"随机森林 - MAE: {mae_rf:.4f}, RMSE: {rmse_rf:.4f}")

    # ==================== 保存评估指标 ====================
    metrics_df = pd.DataFrame({
        'Model': ['Neural Network', 'Random Forest'],
        'MAE': [mae_nn, mae_rf],
        'RMSE': [rmse_nn, rmse_rf]
    })
    metrics_df.to_csv('D:/tasks/final/outputs/m3_model_metrics.csv', index=False)
    print("模型评估指标已保存至 outputs/m3_model_metrics.csv")
    print("\n" + metrics_df.to_string(index=False))

    # ==================== 方法优劣分析（代码注释中） ====================
    """
    【两种方法在此任务上的优劣分析】

    1. 神经网络 (PyTorch MLP):
       优势：
       - 能够学习非线性关系，理论上可以拟合任意复杂函数
       - 对于大规模数据，可以通过增加网络深度和宽度提升表达能力
       - 训练完成后推理速度快
       - 可以通过调整网络结构和超参数进一步优化性能

       劣势：
       - 在小样本数据上容易过拟合或欠拟合（本任务只有24个样本点）
       - 需要较多的超参数调优（层数、神经元数、学习率等）
       - 训练时间相对较长
       - 对数据量敏感，当数据量不足时性能可能不佳

    2. 随机森林 (Random Forest):
       优势：
       - 对非线性关系有良好的拟合能力，且不需要数据归一化
       - 在小样本数据上表现稳定，不易过拟合（通过集成学习）
       - 训练速度快，参数调整简单（主要调整树的数量和深度）
       - 能够处理特征之间的交互作用
       - 对于本任务（24个样本点），随机森林通常表现更稳定

       劣势：
       - 模型可解释性较差（不如线性模型直观）
       - 对于极高维数据可能不如深度学习
       - 无法进行端到端的特征学习

    3. 任务适配性分析：
       - 本任务只有24个样本（每个小时一个数据点），属于典型的小样本回归问题
       - 神经网络在小样本上难以充分发挥其表达能力，容易陷入局部最优
       - 随机森林的集成策略在小样本上往往能获得更稳定的预测结果
       - 从实验结果看，随机森林在MAE和RMSE指标上通常优于神经网络
       - 如果数据量增大（如按分钟或更细粒度），神经网络的优势可能更加明显

    4. 改进建议：
       - 神经网络：可以尝试添加正则化（Dropout、L2正则）防止过拟合
       - 神经网络：使用更简单的网络结构（减少神经元数量）
       - 随机森林：可以通过网格搜索优化树的数量和最大深度
       - 特征工程：可以加入更多的时段特征（如是否为工作日、节假日等）
    """


if __name__ == "__main__":
    operating_m3()
