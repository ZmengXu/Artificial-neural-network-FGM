# !/usr/bin/python
# coding: utf8
# @Time    : 2018-08-04 19:09
# @Author  : Liam
# @Email   : luyu.real@qq.com
# @Software: PyCharm
#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``::::::::::::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'      ::::     .:::::::'::::.
#            .:::'       :::::  .:::::::::' ':::::.
#           .::'        :::::.:::::::::'      ':::::.
#          .::'         ::::::::::::::'         ``::::.
#      ...:::           ::::::::::::'              ``::.
#     ```` ':.          ':::::::::'                  ::::..
#                        '.:::::'                    ':'````..
#                     美女保佑 永无BUG
import torch
#from torch import nn, optim
from torch import nn
import torch.nn.functional as nnFunction

## 自定义一个Net类，继承于torch.nn.Module类
## 这个神经网络的设计是只有一层隐含层，隐含层神经元个数可随意指定
class OneLayer(nn.Module):
    """
    定义了一个简单的两层全连接神经网络，每一层都是线性的
    """
    # Net类的初始化函数
    def __init__(self, n_feature, n_hidden, n_output):
        # 继承父类的初始化函数
        super(OneLayer, self).__init__()
        # 网络的隐藏层创建，名称可以随便起
        self.hidden_layer = nn.Sequential(nn.Linear(n_feature, n_hidden), nn.ReLU(True))#self.hidden_layer = nn.Linear(n_feature, n_hidden)#
        # 输出层(预测层)创建，接收来自隐含层的数据
        self.predict_layer = nn.Linear(n_hidden, n_output)
    # 网络的前向传播函数，构造计算图
    def forward(self, x):
        # 用relu函数处理隐含层输出的结果并传给输出层
        #hidden_result = self.hidden_layer(x)
        x = self.hidden_layer(x)#relu_result = nnFunction.relu(hidden_result)#非常重要，曲面弯曲
        predict_result = self.predict_layer(x)#predict_result = self.predict_layer(relu_result)#
        return predict_result


class simpleNet(nn.Module):
    """
    定义了一个简单的三层全连接神经网络，每一层都是线性的
    """
    # Net类的初始化函数
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
 
class Activation_Net(nn.Module):
    """
    在上面的simpleNet的基础上，在每层的输出部分添加了激活函数
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        """
        这里的Sequential()函数的功能是将网络的层组合到一起。
        """
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
 
class Batch_Net(nn.Module):
    """
    在上面的Activation_Net的基础上，增加了一个加快收敛速度的方法——批标准化
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Batch_Net_5(nn.Module):
    """
    在上面的Activation_Net的基础上，增加了一个加快收敛速度的方法——批标准化
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5, out_dim):
        super(Batch_Net_5, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.BatchNorm1d(n_hidden_4), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5), nn.BatchNorm1d(n_hidden_5), nn.ReLU(True))
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
		
class Batch_Net_5_2(nn.Module):
    """
    在上面的Activation_Net的基础上，增加了一个加快收敛速度的方法——批标准化
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5, out_dim):
        super(Batch_Net_5_2, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5), nn.ReLU(True))
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, out_dim))
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
