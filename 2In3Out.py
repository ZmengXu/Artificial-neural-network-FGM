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

graphsPATH = '02.annGraph/graphs/'
fieldNames = ['C2H3O1-2','C3H4-P','C7H14OOH','CH3CHO','H2O','NC7KET','C2H3','C3H5-A','C7H14O','CH3COCH2','H2','O2','C2H4','C3H5O','C7H15-1','CH3CO','HCCO','OH','C2H5CHO','C3H6','C7H15O2','CH3O2H','HCO','O','C2H5COCH2','C4H6','C7H15','CH3O2','HO2','PC4H9','C2H5O','C4H7O','C7H16','CH3OH','H','C2H5','C4H7','CH2CHO','CH3O','N2','C2H6','C4H8-1','CH2CO','CH3','NC3H7CHO','C2H','C5H10-1','CH2OH','CH4','NC3H7COCH2','C2H2','C3H2','C5H11-1','CH2O','CO2','NC3H7COCH3','C2H3CHO','C3H3','C5H9','CH2-S','CO','NC3H7CO','C2H3CO','C3H4-A','C7H14OOHO2','CH3CHCO','H2O2','NC3H7']


# 训练次数
TRAIN_TIMES = 30000#2040000#68*30000
PATIENCE = 100
PRIDICTION_TOLERANCE = 0.01#1%TOLERANCE of the real data 
# 输入输出的数据维度，这里都是1维
INPUT_FEATURE_DIM = 2
OUTPUT_FEATURE_DIM = len(fieldNames)
# 隐含层中神经元的个数
HIDDEN_1_DIM = 8
HIDDEN_2_DIM = 16
HIDDEN_3_DIM = 32
HIDDEN_4_DIM = 16
HIDDEN_5_DIM = 8

LEARNING_RATE = 0.001




NFGM = []# list of dataframes
for iname in fieldNames:
    speciesI = pd.read_csv('01.orgData/Y-'+iname+'.txt', header=None)
    NFGM.append(speciesI)

# ============================ step 1/6 导入数据 ============================
parser = argparse.ArgumentParser(description='give a filed name and learning rate, for example T')
parser.add_argument('--learningRate', type=float, default = LEARNING_RATE)
args = parser.parse_args()
print(args.learningRate)
LEARNING_RATE = args.learningRate


# Create directory
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


x=Ztarget
y=Ctarget

inputs_np = np.zeros((len(x)*len(y),INPUT_FEATURE_DIM ))
outputs_np = np.zeros((len(x)*len(y),OUTPUT_FEATURE_DIM ))
index = -1
for C in range(len(y)):
    for Z in range(len(x)):    
        index += 1
        inputs_np[index] = np.array([x[Z],y[C]])
        for i, iFGM in enumerate(NFGM):
            outputs_np[index][i] = iFGM.at[C,Z]

x_data = torch.from_numpy(inputs_np).float()
y_data = torch.from_numpy(outputs_np).float()


# randn函数用于生成服从正态分布的随机数
y_data_real = y_data
Min = y_data_real.min(0).values
Max = y_data_real.max(0).values
y_data = (y_data_real-Min)/(Max-Min)

# ============================ step 2/6 选择模型 ============================

# 建立网络
net = netTest.Batch_Net_5_2(INPUT_FEATURE_DIM, HIDDEN_1_DIM, HIDDEN_2_DIM, HIDDEN_3_DIM, HIDDEN_4_DIM, HIDDEN_5_DIM, OUTPUT_FEATURE_DIM)
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

# ============================ step 5/6 模型训练 ============================
for i in range(TRAIN_TIMES):
    # 输入数据进行预测
    prediction = net(x_data)
    #prediction_real = prediction*(Max-Min)+Min
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
    # if LOSS_SQRT < PRIDICTION_TOLERANCE:
        # break    # Lower than tollance break here


# ============================ step 6/6 保存模型 ============================
traced_script_module = torch.jit.trace(net, torch.rand(INPUT_FEATURE_DIM))
traced_script_module.save(graphsPATH+'combinedNet.pt')
torch.save(net.state_dict(), graphsPATH+'combinedNet.pkl')
