# !/usr/bin/python
# coding: utf8
# @Time    : 2020-03-25 19:09

import os
import argparse
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
import netTest


# 训练次数
TRAIN_TIMES = 30000
PATIENCE = 100
PRIDICTION_TOLERANCE = 0.01#1%TOLERANCE of the real data 

# 输入输出的数据维度，这里都是1维
INPUT_FEATURE_DIM = 2
OUTPUT_FEATURE_DIM = 1
# 隐含层中神经元的个数
# HIDDEN_1_DIM = 32
# HIDDEN_2_DIM = 8
# HIDDEN_3_DIM = 8
# HIDDEN_4_DIM = 4
# HIDDEN_5_DIM = 1


# def _initialize_weights(self):
    # print(self.modules())
    # for m in self.modules():
        # print(m)
        # if isinstance(m, nn.Linear):
            # print(m.weight.data.type())
            # input()
            # m.weight.data.fill_(1.0)
            # init.xavier_uniform_(m.weight, gain=1)

# torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')


# ============================ step 1/6 导入数据 ============================
parser = argparse.ArgumentParser(description='give a filed name and learning rate, for example T')
parser.add_argument('--fieldName', type=str, default = None)
parser.add_argument('--learningRate', type=float, default = 0.001)
args = parser.parse_args()
print(args.fieldName)
print(args.learningRate)
LEARNING_RATE = args.learningRate

fieldName = args.fieldName
dataFile = '01.orgData/Y_'+fieldName+'.txt'
animationPATH = '02.annGraph/animations/'+fieldName#+'/LEARNING_RATE'+str(args.learningRate)
#animationPATH = '02.annGraph/animations/'+fieldName
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
# The number of Z and PV for creating the input array
nZ=1001
nPV=501
Ztarget = np.linspace(0,1,nZ)
Ctarget = np.square(np.linspace(0,1,nPV))
FGM = pd.read_csv(dataFile, header=None)

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
Min = y_data_real.min()
Max = y_data_real.max()
y_data = (y_data_real-Min)/(Max-Min)#*(1 + PRIDICTION_TOLERANCE*torch.randn(y_data.size()))

#y_data = y_data_real#*(1 + PRIDICTION_TOLERANCE*torch.randn(y_data.size()))


# ============================ step 2/6 选择模型 ============================

# def init_params(net):
    # '''Init layer parameters.'''
    # for m in net.modules():
        # if isinstance(m, torch.nn.Conv2d):
            # torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
            # # if m.bias:
                # # torch.nn.init.constant(m.bias, 0)
        # elif isinstance(m, torch.nn.BatchNorm2d):
            # torch.nn.init.constant(m.weight, 1)
            # #torch.nn.init.constant(m.bias, 0)
        # elif isinstance(m, torch.nn.Linear):
            # torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')#torch.nn.init.normal(m.weight, std=1e-3)
            # m.weight.data = m.weight.data/500
            # # if m.bias:
                # # torch.nn.init.constant(m.bias, False)
                
              
# 建立网络
net = netTest.Batch_Net_5_2(INPUT_FEATURE_DIM, 8, 16, 32, 16, 8, OUTPUT_FEATURE_DIM)
# init_params(net)
print(net)



# ============================ step 3/6 选择优化器   =========================

optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)


# ============================ step 4/6 选择损失函数 =========================
# 定义一个误差计算方法
# Mean Square Error (MSE), MSELoss(), L2 loss
# Mean Absolute Error (MAE), MAELoss(), L1 Loss
loss_func = torch.nn.MSELoss()
INITIAL_LOSS = loss_func(0*y_data, y_data)
LOSS_TOLERANCE = loss_func(0*y_data, PRIDICTION_TOLERANCE*y_data).data.numpy()
print('LOSS_TOLERANCE',LOSS_TOLERANCE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, threshold=PATIENCE/TRAIN_TIMES*LOSS_TOLERANCE, patience=PATIENCE)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_wireframe(xx_2D, yy_2D, zz_2D, color = 'blue',linewidth=1, rstride=10, cstride=10)
# The second traning

# ============================ step 5/6 模型训练 ============================
for i in range(TRAIN_TIMES):
    # 输入数据进行预测
    prediction = net(x_data)
    prediction_real = prediction*(Max-Min)+Min
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
    LOSS_SQRT = np.sqrt(loss.data.numpy()/INITIAL_LOSS.data.numpy())
    print("Iteration : ",'%05d'%i, "\tLearningRate : {:.3e}\tLoss: {:.4e}\tRelativeError:{:.5e}".format( optimizer.param_groups[0]['lr'], loss.data.numpy(), LOSS_SQRT ))  
    if LOSS_SQRT < PRIDICTION_TOLERANCE:
        break    # Lower than tollance break here
    if LOSS_SQRT < PRIDICTION_TOLERANCE:
        break    # Lower than tollance break here
    if i % 50 == 0:
        # 实时预测的曲面
        plotPrid = np.reshape(prediction_real.data.numpy()[:,0],(len(y),len(x)))
        pridSurf = ax.plot_wireframe(xx_2D, yy_2D, plotPrid, color = 'red',linewidth=1, rstride=50, cstride=50)
        plt.savefig(animationPATH+'/'+str('%05d'%i)+'.png', dpi=96)
        # 清空上一次显示结果
        pridSurf.remove()

# y_data = y_data_real
# for m in net.modules():
    # if isinstance(m, torch.nn.Linear):
        # m.weight.data = m.weight.data/5000

# loss_func = torch.nn.MSELoss()
# INITIAL_LOSS = loss_func(0*y_data, y_data)
# LOSS_TOLERANCE = loss_func(0*y_data, PRIDICTION_TOLERANCE*y_data).data.numpy()
# print('LOSS_TOLERANCE',LOSS_TOLERANCE)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, threshold=PATIENCE/TRAIN_TIMES*LOSS_TOLERANCE, patience=PATIENCE)


# # fig = plt.figure()
# # ax = fig.gca(projection='3d')
# # ax.plot_wireframe(xx_2D, yy_2D, zz_2D, color = 'blue',linewidth=1, rstride=10, cstride=10)
# # # The second traning

# for i in range(TRAIN_TIMES):
    # # 输入数据进行预测
    # prediction = net(x_data)
    # # 计算预测值与真值误差，注意参数顺序问题
    # # 第一个参数为预测值，第二个为真值
    # loss = loss_func(prediction, y_data)#*100/firstLoss
    # # Change the learning rate
    # #scheduler.step()
    # scheduler.step(loss)
    # # 开始优化步骤
    # # 每次开始优化前将梯度置为0
    # optimizer.zero_grad()
    # # 误差反向传播
    # loss.backward()
    # optimizer.step()
    # # 按照最小loss优化参数
    # # 可视化训练结果
    # LOSS_SQRT = np.sqrt(loss.data.numpy()/INITIAL_LOSS.data.numpy())
    # print("Iteration : ",'%05d'%i, "\tLearningRate : {:.3e}\tLoss: {:.4e}\tRelativeError:{:.5e}".format( optimizer.param_groups[0]['lr'], loss.data.numpy(), LOSS_SQRT ))  
    # if LOSS_SQRT < PRIDICTION_TOLERANCE:
        # break    # Lower than tollance break here
    # if i % 50 == 0:
        # # 实时预测的曲面
        # plotPrid = np.reshape(prediction.data.numpy()[:,0],(len(y),len(x)))
        # pridSurf = ax.plot_wireframe(xx_2D, yy_2D, plotPrid, color = 'red',linewidth=1, rstride=50, cstride=50)
        # plt.savefig(animationPATH+'/'+str('%05d'%i)+'.png', dpi=96)
        # # 清空上一次显示结果
        # pridSurf.remove()


# ============================ step 6/6 保存模型 ============================
traced_script_module = torch.jit.trace(net, torch.rand(2))
traced_script_module.save(graphsPATH+'/'+fieldName+'.pt')
torch.save(net.state_dict(), graphsPATH+'/'+fieldName+'.pkl')
