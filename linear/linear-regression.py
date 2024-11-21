import torch
from d2l import torch as d2l
from torch import nn
from torch.utils import data

# 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# 读取数据集
def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 定义模型
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数
# normal_()方法用于将张量用正态分布(高斯分布)的值填充
# 参数分别是均值(mean)=0和标准差(std)=0.01
# 这里用于随机初始化线性层的权重参数
net[0].weight.data.normal_(0, 0.01)
# fill_()方法将张量中的所有元素填充为指定的值
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        loss_val = loss(net(X), y)
        # 在每次更新参数前需要清零梯度
        # 因为PyTorch会累积梯度,如果不清零,梯度会累加到之前的梯度上
        trainer.zero_grad()

        # backward()计算损失相对于参数的梯度
        loss_val.backward()

        # step()根据梯度更新模型参数
        # 使用之前定义的优化器(SGD)和学习率来更新参数
        trainer.step()

    # 在每个epoch结束时计算整个数据集上的损失
    loss_val = loss(net(features), labels)
    print(f"epoch {epoch + 1}, loss {loss_val:f}")

# 比较真实参数和通过训练学到的参数来评估训练的成功程度
print(f"w的估计误差: {true_w - net[0].weight.data}")
print(f"b的估计误差: {true_b - net[0].bias.data}")
