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



# 训练次数
TRAIN_TIMES = 30000
PRIDICTION_TOLERANCE = 0.01#1%TOLERANCE of the real data 
# 学习率，越大学的越快，但也容易造成不稳定，准确率上下波动的情况
LEARNING_RATE = 0.001
# 输入输出的数据维度，这里都是1维
INPUT_FEATURE_DIM = 2
OUTPUT_FEATURE_DIM = 1
# 隐含层中神经元的个数
# HIDDEN_1_DIM = 32
# HIDDEN_2_DIM = 8
# HIDDEN_3_DIM = 8
# HIDDEN_4_DIM = 4
# HIDDEN_5_DIM = 1


# The number of Z and PV for creating the input array
nZ=1001
nPV=501

# ============================ step 1/6 导入数据 ============================
fieldName = args.fileName
#animationPATH = '02.annGraph/animations/'+fieldName+'/LEARNING_RATE'+str(LEARNING_RATE)
animationPATH = '02.annGraph/animations/'+fieldName
graphsPATH = '02.annGraph/graphs/'+fieldName
# Create directory
if not os.path.exists(animationPATH):
    os.mkdir(animationPATH)
    print("Directory " , animationPATH ,  " Created ")
else:    
    print("Directory " , animationPATH ,  " already exists")

if not os.path.exists(graphsPATH):
    os.mkdir(graphsPATH)
    print("Directory " , graphsPATH ,  " Created ")
else:    
    print("Directory " , graphsPATH ,  " already exists")




# 数据构造
# 这里x_data、y_data都是tensor格式，在PyTorch0.4版本以后，也能进行反向传播
# 所以不需要再转成Variable格式了
# linspace函数用于生成一系列数据
# unsqueeze函数可以将一维数据变成二维数据，在torch中只能处理二维数据

Ztarget = np.linspace(0,1,nZ)
Ctarget = np.square(np.linspace(0,1,nPV))
FGM = pd.read_csv('01.orgData/'+fieldName+'.txt',header=None)



x=Ztarget
y=Ctarget


inputs_np = np.zeros((len(x)*len(y),2 ))
outputs_np = np.zeros((len(x)*len(y),1 ))
index = -1
for C in range(len(y)):
    for Z in range(len(x)):    
        index += 1
        inputs_np[index] = np.array([x[Z],y[C]])
        outputs_np[index] = np.array([FGM.at[C,Z]])



xx_2D, yy_2D = np.meshgrid(x, y)
zz_2D = np.reshape(outputs_np,(len(y),len(x)))


x_data = torch.from_numpy(inputs_np).float()
y_data = torch.from_numpy(outputs_np).float()


# randn函数用于生成服从正态分布的随机数
y_data_real = y_data
y_data = y_data_real#*(1 + 0.01*torch.randn(y_data.size()))

# 学习率，越大学的越快，但也容易造成不稳定，准确率上下波动的情况
#LEARNING_RATE = 0.1*(y_data.max()-y_data.min())
# residual = (0.01*y_data)

#math.sqrt(sum([x ** 2 for x in records]) / len(records))
# ============================ step 2/6 选择模型 ============================

# 建立网络
#net = netTest.OneLayer(n_feature=INPUT_FEATURE_DIM, n_hidden=NEURON_NUM, n_output=OUTPUT_FEATURE_DIM)
#net = netTest.simpleNet(INPUT_FEATURE_DIM, 10, 3, OUTPUT_FEATURE_DIM)
#net = netTest.Activation_Net(INPUT_FEATURE_DIM, 32, 2, OUTPUT_FEATURE_DIM)
#net = netTest.Batch_Net(INPUT_FEATURE_DIM, 32, 3, OUTPUT_FEATURE_DIM)
#net = netTest.Batch_Net_5(INPUT_FEATURE_DIM, 32, 5, 3, 3, 3, OUTPUT_FEATURE_DIM)
net = netTest.Batch_Net_5_2(INPUT_FEATURE_DIM, 8, 16, 32, 16, 8, OUTPUT_FEATURE_DIM)

print(net)


# ============================ step 3/6 选择优化器   =========================

optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
#optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, amsgrad=True)
# sum of the initial value
#y_data_sum = torch.sum(y_data)
#optimizer = torch.optim.Adagrad(net.parameters(), lr=LEARNING_RATE, initial_accumulator_value=y_data_sum)

# Learning rate adapting
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)


#scheduler.get_lr()[0]

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50)

#optimizer = torch.optim.SGD(net.parameters(), lr=0.002, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)

#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
#
#>>> # Assuming optimizer uses lr = 0.05 for all groups
#>>> # lr = 0.05     if epoch < 30
#>>> # lr = 0.005    if 30 <= epoch < 80
#>>> # lr = 0.0005   if epoch >= 80
#>>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
#StepLR, MultiStepLR, ExponentialLR or CosineAnnealingLR scheduler 
#scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
#optimizer.param_groups[0]['lr']

#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, last_epoch=-1)
#scheduler.step()
# ============================ step 4/6 选择损失函数 =========================
# 定义一个误差计算方法
# Mean Square Error (MSE), MSELoss(), L2 loss
# Mean Absolute Error (MAE), MAELoss(), L1 Loss
loss_func = torch.nn.MSELoss()

LOSS_TOLERANCE = loss_func(0*y_data, PRIDICTION_TOLERANCE*y_data)

print("LOSS_TOLERANCE : ",'%.4e'%LOSS_TOLERANCE)
# 输入数据进行预测
#firstLoss = loss_func(net(x_data), y_data).data.numpy()
#firstLoss = np.linalg.norm(firstLoss)

# ============================ step 5/6 模型训练 ============================
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_wireframe(xx_2D, yy_2D, zz_2D, color = 'blue',linewidth=1, rstride=10, cstride=10)

for i in range(TRAIN_TIMES):
    # 输入数据进行预测
    prediction = net(x_data)
    # 计算预测值与真值误差，注意参数顺序问题
    # 第一个参数为预测值，第二个为真值
    loss = loss_func(prediction, y_data)#*100/firstLoss
    # Change the learning rate
    #scheduler.step()
    scheduler.step(loss)
    # 开始优化步骤
    # 每次开始优化前将梯度置为0
    optimizer.zero_grad()
    # 误差反向传播
    loss.backward()
    optimizer.step()
    # 按照最小loss优化参数
    # 可视化训练结果
    print("Iteration : ",'%05d'%i, "\tLearningRate : {:.4e}\tLoss: {:.5e}".format( optimizer.param_groups[0]['lr'], loss.data.numpy() ))  
    if loss < LOSS_TOLERANCE:
        break    # Lower than tollance break here
    if i % 50 == 0:
        # 清空上一次显示结果
        #ax.cla()
        # 无误差真值网格
        #surf1 = ax.plot_wireframe(xx_2D, yy_2D, zz_2D, color = 'blue',linewidth=1, rstride=50, cstride=50)
        #ax.plot_trisurf(x_data.numpy()[:,0], x_data.numpy()[:,1], y_data.numpy()[:,0], color = 'red', linewidth=2)
        # 有误差散点
        #ax.scatter(x_data.numpy()[:,0], x_data.numpy()[:,1], y_data.numpy()[:,0],color = 'orange')
        # 实时预测的曲面
        plotPrid = np.reshape(prediction.data.numpy()[:,0],(len(y),len(x)))
        #pridSurf = ax.plot_surface(xx_2D, yy_2D, plotPrid, color = 'red',linewidth=1, rstride=50, cstride=50)
        pridSurf = ax.plot_wireframe(xx_2D, yy_2D, plotPrid, color = 'red',linewidth=1, rstride=50, cstride=50)
        #ax.plot_trisurf(x_data.numpy()[:,0], x_data.numpy()[:,1], prediction.data.numpy()[:,0], color = 'red',linewidth=2)
        #ax.text('Time=%d Loss=%.4f' % (i, loss.data.numpy()), fontdict={'size': 15, 'color': 'red'})
        #plt.pause(0.1)
        plt.savefig(animationPATH+'/'+str('%05d'%i)+'.png', dpi=96)
        # 清空上一次显示结果
        pridSurf.remove()



# ============================ step 6/6 保存模型 ============================
traced_script_module = torch.jit.trace(net, torch.rand(2))
traced_script_module.save(graphsPATH+'/'+fieldName+'.pt')
torch.save(net.state_dict(), graphsPATH+'/'+fieldName+'.pkl')
# 这种方式将会提取整个神经网络, 网络大的时候可能会比较慢.
#print torch.load('net_params.pkl')
# 打印输出我们会发现，上面只保存了模型参数
 
# 将保存的参数复制到 net,这种方式将会提取所有的参数, 然后再放到你的新建网络中.
#net.load_state_dict(torch.load('net_params.pkl'))	