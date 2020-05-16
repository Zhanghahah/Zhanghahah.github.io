---
layout: post
title:  "Pytorch中常用的一些方法"
subtitle:	
date:   2020/5/16 13:03:42
header-img: "img/post-think-try-write.jpg"
---



# pytorch中常用的一些方法：



```python
acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item() 
# dim = 1 每一行的最大列索引下标

```

### batch size/ epoch/ iteration 的关系

```
一个epoch指代**所有的数据**送入网络中完成一次前向计算及反向传播的过程。

batch就是每次送入网络中训练的一部分数据，batch size就是每个batch中训练样本的数量。

iterations就是完成一次epoch所需的batch个数。比如，训练集一共有5000个数据，batch size为500，则iterations=10，epoch是等于1
```

### 实现step:

```python
for X, y in train_iter:
    X = X.to(device) # 把测试数据都放到GPU上
	y = y.to(device)
	y_hat = net(X)   # 前向传播
	l = loss(y_hat, y) #计算 loss
	optimizer.zero_grad() # 梯度清0
	l.backward() # 反向传播
	optimizer.step()  # 更新参数
```

### dropout 应该加在哪一层，有什么说法

```
Dropout 层一般加在全连接层之后，防止过拟合，提升模型泛化能力。而很少见到卷积层后接Dropout （原因主要是 卷积参数少，不易过拟合）
```

### 几种网络结构的对比

```python
# lenet
self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
# alexnet
self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, 
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
         # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),  #注意 dropout 加在激活函数之后
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10),
        )
```

```python
 net.add_module("fc", nn.Sequential(d2l.FlattenLayer(),
                                 nn.Linear(fc_features, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, 10)
                                ))
    #VGG net
```

### `modules()`,`named_modules()`,`children()`,`named_children()`的区别

`1.modules()`

```python
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3)
        self.conv2 = nn.Conv2d(64,64,3)
        self.maxpool1 = nn.MaxPool2d(2,2)

        self.features = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(64,128,3)),
            ('conv4', nn.Conv2d(128,128,3)),
            ('relu1', nn.ReLU())
        ]))

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.features(x)

        return x

m = Model()
for idx, m in enumerate(m.modules()):
    print(idx,"-",m)

    
    
results:
 0 - Model(
  (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
  (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (features): Sequential(
    (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
    (conv4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
    (relu1): ReLU()
  )
)
1 - Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
2 - Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
3 - MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
4 - Sequential(
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
  (conv4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
  (relu1): ReLU()
)
5 - Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
6 - Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
7 - ReLU()`
    
```

递归的返回网络结构

`2.parameters()`

```
parameters()返回网络的所有参数，主要是提供给optimizer用的。而要取得网络某一层的参数或者参数进行一些特殊的处理（如fine-tuning)，则使用named_parameters()更为方便些。named_parameters()返回参数的名称及参数本身，可以按照参数名对一些参数进行处理。
named_parameters返回的是键值对，k为参数的名称 ，v为参数本身
```

`3.children()/ named_children()`

```
children()返回模块本身，不包含名字
name_children() 返回模块名字和模块， 二者都不会递归子模块
```

`4.不要把输入通道数和像素块大小搞混`

```python 
输入通道数 = feature map 的个数
像素块矩阵  =  feature map 的大小 指有多少个像素
ex: X.torh.randn([1,1,224,224])

0 output shape:  torch.Size([1, 96, 54, 54])  #  54 54 指的是像素大小 96才是通道数量
1 output shape:  torch.Size([1, 96, 26, 26])  # 即feature map的个数
2 output shape:  torch.Size([1, 256, 26, 26])
3 output shape:  torch.Size([1, 256, 12, 12])
4 output shape:  torch.Size([1, 384, 12, 12])
5 output shape:  torch.Size([1, 384, 5, 5])
6 output shape:  torch.Size([1, 384, 5, 5])
7 output shape:  torch.Size([1, 10, 5, 5])
8 output shape:  torch.Size([1, 10, 1, 1])
9 output shape:  torch.Size([1, 10])
```

`5. batch normalization`

```python
#对卷积层来说，批量归一化发生在卷积计算之后、激活函数之前。
#如果卷积计算输出多个通道，需要对BN这些通道分别做BN，每个通道都有独立的拉伸和偏移，都为标量。
#假设一个batch size有 m个样本，在每个通道上输出的都是p*q。那么需要对 m*p*q个元素同时做BN，并且使用相同的均值和方差，即通道中所有 m*p*q个元素的均值和方差。

# ex：  nn.BatchNorm2d(64), 64个通道
mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(6),
            nn.Sigmoid(),
#对全连接层：
mean = X.mean(dim=0)
var = ((X - mean) ** 2).mean(dim=0)

nn.Linear(16*4*4, 120),
            nn.BatchNorm1d(120),
            nn.Sigmoid(),
BNz在training和test时也和dropout一样不相同：

实现一个BN
		if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。这里我们需要保持
            # X的形状以便后面可以做广播运算
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        # 训练模式下用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 拉伸和偏移
```

