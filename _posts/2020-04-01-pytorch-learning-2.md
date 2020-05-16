# pytorch 中的一些方法2

`1. 对Resnet的理解`：

`从宏观上看是一个weight layer + 激活函数 + weight layer的结构，但还需理解weight layer中的结构特性`

`可以具体拆分开是`

```python
# Resnet block结构：conv + BN
conv1 + BN1 = weight layer1
conv2 + BN2 = weight layer2
x  -> conv1 + BN1 -> Relu  -> conv2 + BN2 =======  y
output: relu(x + y) 	
# trick 是否用 1*1的卷积层

# Resnet的 改良结构： BN + Relu + conv
def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels), 
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    return blk

```

```python
class Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)
```

`2. Denset`

```python
class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels # 计算输出通道数

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维上将输入和输出连结
        return X
```

`3. torchvision.transforms中的一些trick` `我比较关注RandomResizedCrop的一些参数`

`shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))` 

scale 随机裁剪⼀⽚⾯积为原⾯积 10% 到 100% 的区域，ratio 其宽和⾼的⽐例在 0.5 和 2 之间，然后再将⾼宽缩放到 200 像素⼤小 。