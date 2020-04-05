# !/usr/bin/python
# coding: utf8
# @Time    : 2020-03-25 19:09

import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F

import netTest

import os
import argparse
parser = argparse.ArgumentParser(description='give a filed name, for example T')
parser.add_argument('--fileName', type=str, default = None)
args = parser.parse_args()
print(args.fileName)


fieldName = args.fileName
# Create directory
if not os.path.exists('02.annGraph/animations/'+fieldName):
    os.mkdir('02.annGraph/animations/'+fieldName)
    print("Directory " , fieldName ,  " Created ")
else:    
    print("Directory " , fieldName ,  " already exists")



# 训练次数
TRAIN_TIMES = 3000
# 输入输出的数据维度，这里都是1维
INPUT_FEATURE_DIM = 2
OUTPUT_FEATURE_DIM = 1
# 隐含层中神经元的个数
# HIDDEN_1_DIM = 32
# HIDDEN_2_DIM = 8
# HIDDEN_3_DIM = 8
# HIDDEN_4_DIM = 4
# HIDDEN_5_DIM = 1
# 学习率，越大学的越快，但也容易造成不稳定，准确率上下波动的情况
LEARNING_RATE = 0.1

# 数据构造
# 这里x_data、y_data都是tensor格式，在PyTorch0.4版本以后，也能进行反向传播
# 所以不需要再转成Variable格式了
# linspace函数用于生成一系列数据
# unsqueeze函数可以将一维数据变成二维数据，在torch中只能处理二维数据

nZ=1001
nPV=501

Ztarget = np.linspace(0,1,nZ)
Ctarget = np.square(np.linspace(0,1,nPV))
FGM = pd.read_csv('01.orgData/'+fieldName+'.txt',header=None)

inputs_np = np.zeros((len(Ztarget)*len(Ctarget),2 ))
outputs_np = np.zeros((len(Ztarget)*len(Ctarget),1 ))
index = -1
for C in range(len(Ctarget)):
    for Z in range(len(Ztarget)):    
        index += 1
        inputs_np[index] = np.array([Ztarget[Z],Ctarget[C]])
        outputs_np[index] = np.array([FGM.at[C,Z]])


x=Ztarget
y=Ctarget
xx_2D, yy_2D = np.meshgrid(x, y)
zz_2D = np.reshape(outputs_np,(y.size,x.size))


x_data = torch.from_numpy(inputs_np).float()
y_data = torch.from_numpy(outputs_np).float()


# randn函数用于生成服从正态分布的随机数
y_data_real = y_data
y_data = y_data_real + 0.01*torch.randn(y_data.size())


# 建立网络
#net = netTest.OneLayer(n_feature=INPUT_FEATURE_DIM, n_hidden=NEURON_NUM, n_output=OUTPUT_FEATURE_DIM)
#net = netTest.simpleNet(INPUT_FEATURE_DIM, 10, 3, OUTPUT_FEATURE_DIM)
#net = netTest.Activation_Net(INPUT_FEATURE_DIM, 32, 2, OUTPUT_FEATURE_DIM)
#net = netTest.Batch_Net(INPUT_FEATURE_DIM, 32, 3, OUTPUT_FEATURE_DIM)
#net = netTest.Batch_Net_5(INPUT_FEATURE_DIM, 32, 5, 3, 3, 3, OUTPUT_FEATURE_DIM)
net = netTest.Batch_Net_5_2(INPUT_FEATURE_DIM, 8, 16, 32, 16, 8, OUTPUT_FEATURE_DIM)

print(net)

# 训练网络
# 这里也可以使用其它的优化方法
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
# 定义一个误差计算方法
loss_func = torch.nn.MSELoss()

# 输入数据进行预测
firstLoss = loss_func(net(x_data), y_data).data.numpy()
firstLoss = np.linalg.norm(firstLoss)


fig = plt.figure()
ax = fig.gca(projection='3d')
for i in range(TRAIN_TIMES):
    # 输入数据进行预测
    prediction = net(x_data)
    # 计算预测值与真值误差，注意参数顺序问题
    # 第一个参数为预测值，第二个为真值
    loss = loss_func(prediction, y_data)*100/firstLoss
    # 开始优化步骤
    # 每次开始优化前将梯度置为0
    optimizer.zero_grad()
    # 误差反向传播
    loss.backward()
    # 按照最小loss优化参数
    optimizer.step()
    # 可视化训练结果
    print("Iteration : {:.0f} Loss: {:.4f}".format( i, loss.data.numpy() ))  
    if i % 50 == 0:
        # 清空上一次显示结果
        ax.cla()
        # 无误差真值网格
        ax.plot_wireframe(xx_2D, yy_2D, zz_2D, color = 'blue',linewidth=1)
        #ax.plot_trisurf(x_data.numpy()[:,0], x_data.numpy()[:,1], y_data.numpy()[:,0], color = 'red', linewidth=2)
        # 有误差散点
        #ax.scatter(x_data.numpy()[:,0], x_data.numpy()[:,1], y_data.numpy()[:,0],color = 'orange')
        # 实时预测的曲面
        ax.plot_trisurf(x_data.numpy()[:,0], x_data.numpy()[:,1], prediction.data.numpy()[:,0], color = 'red',linewidth=2)
        #ax.text('Time=%d Loss=%.4f' % (i, loss.data.numpy()), fontdict={'size': 15, 'color': 'red'})
        #plt.pause(0.1)
        plt.savefig('02.annGraph/animations/'+fieldName+'/'+str('%03d'%i)+'.png', dpi=96)

torch.save(net.state_dict(), '02.annGraph/graphs/'+fieldName+'.pkl')

traced_script_module = torch.jit.trace(net, torch.rand(2))
traced_script_module.save('02.annGraph/graphs/'+fieldName+'.pt')
# 这种方式将会提取整个神经网络, 网络大的时候可能会比较慢.
#print torch.load('net_params.pkl')
# 打印输出我们会发现，上面只保存了模型参数
 
# 将保存的参数复制到 net,这种方式将会提取所有的参数, 然后再放到你的新建网络中.
#net.load_state_dict(torch.load('net_params.pkl'))	