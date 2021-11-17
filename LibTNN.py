'''
    LibTNN
    Author: YB.Li
    本库包含一下层与函数：
    1. CP_TRL(CP Tensor Regression Layer)           CP分解张量回归层
    2. Tucker_TRL(Tucker Tensor Regression Layer)   Tucker分解张量回归层
    3. TCL(Tensor Contraction Layer)                张量收缩层
    4. CP_CNN(CP SPEED-UP CNN)                      CP分解加速的CNN层
    5. Tucker_CNN(Tucker SPEED-UP CNN)              Tucker分解加速的CNN层
    6. cp_decomposition_conv_layer                  使用CP分解加速CNN层
    7. tucker_decomposition_conv_layer              使用Tucker分解加速的CNN层
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import tensorly as tl
from tensorly.tenalg import inner
from tensorly.decomposition import partial_tucker
from tensorly.decomposition import parafac

tl.set_backend('pytorch')


class CP_TRL(nn.Module):
    
    '''
    CP_TRL(CP Tensor Regression Layer) 层的实现
    实现思想:
    对于一个大小为 (batch_size, m1, m2, m3) 大小的激活张量，网络回归希望得到一个大小为 (batch_size, class_num) 的输出
    为了得到和预期输出，可以使用一个大小为 (m1, m2, m3, class_num) 的回归权重张量，采用广义内积 (Generalized inner-product) 的形式，得到输出
    回归权重张量的大小过大，不便于计算，可以使用 CP 分解的思想，使用秩 R 的 CP 分解
    权重系数矩阵大小为 (R, 1)，分解矩阵大小分别为 (m1, R) (m2, R) (m3, R) (class_num, R)，简化计算
    '''
    
    def __init__(self, input_size: tuple, output_size: tuple, rank: int, verbose=1, **kwargs):
        
        '''
            input_size: tuple   为输入激活张量的大小 (batch_size, m1, m2, m3)，其第一个参数 batch_size 无效
            output_size: tuple  为输出矩阵的大小 (batch_size, class_num)，其第一个参数 batch_size 无效
            rank: int           为回归张量 CP 分解的秩
        '''
        
        super(CP_TRL, self).__init__(**kwargs)
        self.verbose = verbose
        
        # 参数列表化
        self.input_size = list(input_size)
        self.output_size = list(output_size)
        
        # 输出类别数
        self.n_outputs = int(np.prod(output_size[1:]))
        
        self.bias = nn.Parameter(tl.zeros(1), requires_grad=True)               # 偏置
        self.weight = nn.Parameter(tl.ones(rank, ), requires_grad=True)         # CP 分解权重矩阵
        
        weight_size = list(self.input_size[1:]) + list(self.output_size[1:])    # 分解矩阵 dim=0 方向大小 (m1, m2, m3, class_num)
        
        self.factors = []   # 初始化分解矩阵队列
        for index, in_size in enumerate(weight_size):
            # 分解矩阵大小分别为 (m1, R) (m2, R) (m3, R) (class_num, R)
            self.factors.append(nn.Parameter(tl.zeros((in_size, rank))))
            self.register_parameter('factor_{}'.format(index), self.factors[index])    # 注册参数
        
        # 重新分布分解矩阵
        for f in self.factors:
            f.data.uniform_(-0.1, 0.1)
            
    def forward(self, x):
        regression_weights = tl.cp_to_tensor((self.weight, self.factors))
        return inner(x, regression_weights, n_modes=tl.ndim(x)-1) + self.bias   # 广义内积


class Tucker_TRL(nn.Module):
    
    '''
    TRL(Tucker Tensor Regression Layer) 层的实现
    实现思想:
    对于一个大小为 (batch_size, m1, m2, m3) 大小的激活张量，网络回归希望得到一个大小为 (batch_size, class_num) 的输出
    为了得到和预期输出，可以使用一个大小为 (m1, m2, m3, class_num) 的回归权重张量，采用广义内积 (Generalized inner-product) 的形式，得到输出
    回归权重张量的大小过大，不便于计算，可以使用 Tucker 分解的思想，使用秩 (a, b, c, d) 的 Tucker 分解
    核心张量大小为 (a, b, c, d)，分解矩阵大小分别为 (m1, a) (m2, b) (m3, c) (class_num, d)，简化计算
    '''
    
    def __init__(self, input_size: tuple, output_size: tuple, ranks: tuple, verbose=1, **kwargs):
        
        '''
            input_size: tuple   为输入激活张量的大小 (batch_size, m1, m2, m3)，其第一个参数 batch_size 无效
            output_size: tuple  为输出矩阵的大小 (batch_size, class_num)，其第一个参数 batch_size 无效
            ranks: tuple        为回归张量 Tucker 分解核心张量大小
        '''
        
        super(Tucker_TRL, self).__init__(**kwargs)
        
        self.ranks = list(ranks)
        self.verbose = verbose

        # 参数列表化
        self.input_size = list(input_size)
        self.output_size = list(output_size)
        
        # 输出类别数
        self.n_outputs = int(np.prod(output_size[1:]))

        self.core = nn.Parameter(tl.zeros(self.ranks), requires_grad=True)      # 初始化 Tucker 回归核心张量（权重）
        self.bias = nn.Parameter(tl.zeros(1), requires_grad=True)               # 偏置
        weight_size = list(self.input_size[1:]) + list(self.output_size[1:])    # 分解矩阵 dim=0 方向大小 (m1, m2, m3, class_num)

        self.factors = []   # 初始化分解矩阵队列
        for index, (in_size, rank) in enumerate(zip(weight_size, ranks)):       # zip 包装成元组，并使用枚举
            # (m1, a) (m2, b) (m3, c) (class_num, d)
            self.factors.append(nn.Parameter(tl.zeros((in_size, rank)), requires_grad=True))
            self.register_parameter('factor_{}'.format(index), self.factors[index])
        
        self.core.data.uniform_(-0.1, 0.1)      # 重新分布核心张量
        for f in self.factors:
            f.data.uniform_(-0.1, 0.1)          # 重新分布分解张量

    def forward(self, x):           # 前向传播
        regression_weights = tl.tucker_to_tensor((self.core, self.factors))
        return inner(x, regression_weights, n_modes=tl.ndim(x)-1) + self.bias


class TCL(nn.Module):
    
    '''
    TCL(Tensor Contraction Layer)层的实现
    实现思想：
    对于一个大小为 (batch_size, m1, m2, m3) 大小的激活张量，希望得到一个大小为 (batch_size, n1, n2, n3) 的输出作为下一层的输入
    可以使用模态积的方法，使用三个大小为 (n1, m1) (n2, m2) (n3, m3) 的矩阵，分别进行 mode-1 mode-2 mode-3 的模态积
    可以用来代替展平层和全连接层
    '''
    
    def __init__(self, input_size: tuple, output_size: tuple, verbose=1, **kwargs):
        
        '''
            input_size: tuple   为输入激活张量的大小 (batch_size, m1, m2, m3)，其第一个参数 batch_size 无效
            output_size: tuple  为输出矩阵的大小 (batch_size, class_num)，其第一个参数 batch_size 无效
        '''
        
        super(TCL, self).__init__(**kwargs)
        self.verbose = verbose
        
        # 设定层的输入输出大小
        self.input_size = list(input_size)
        self.output_size = list(output_size)
        
        # 获取权重张量 x 与 y 方向上的大小
        weight_size_x = list(self.output_size[1:])
        weight_size_y = list(self.input_size[1:])
        
        # 建立 factors 队列
        self.factors = []
        
        # 生成权重矩阵队列
        for index, (size_x, size_y) in enumerate(zip(weight_size_x, weight_size_y)):
            self.factors.append(nn.Parameter(tl.zeros((size_x, size_y)), requires_grad=True))   # 生成权重矩阵
            self.register_parameter('factor_{}'.format(index), self.factors[index])             # 注册参数
        
        # 权重矩阵参数随机分布
        for f in self.factors:
            f.data.uniform_(-0.1, 0.1)
            
    def forward(self, x):
        
        # 从 mode-1 位置进行模态点乘
        for index, factor in enumerate(self.factors):
            x = tl.tenalg.mode_dot(x, factor, mode=index+1)
        
        return x


def cp_decomposition_conv_layer(layer, rank):
    
    """ 
        layer: torch.nnConv2d   pytorch二维卷积层
        rank:  int              卷积层张量 CP 分解秩
        
        Gets a conv layer and a target rank, 
        returns a nn.Sequential object with the decomposition
        接受一个给定的卷积层与目标 CP 分解秩
        返回一个经过分解的 nn.Sequential 对象
    """
    
    # Perform CP decomposition on the layer weight tensorly.
    # 使用 tensorly 对卷积核进行 CP分解
    # bug !!!!!!
    # 注：last 表示原始卷积核张量上第一维（卷积核个数维）的分量
    #    first 表示原始卷积核张量上第二维（输入通道数）的分量
    #    vertical 表示表示原始卷积核张量上第二维（输入图 x 方向）的分量
    #    vertical 表示表示原始卷积核张量上第二维（输入图 y 方向）的分量
    weights, [last, first, vertical, horizontal] = \
        parafac(layer.weight.data, rank=rank, init='svd')

    # S(卷积核数目) to R(CP分解秩) 的通道压缩逐点卷积（pointwise convolution）
    pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1, stride=1, padding=0, 
            dilation=layer.dilation, bias=False)
    # in_channels=first.shape[0]=S
    # out_channels=first.shape[1]=R
    # kernel_size=1 代表逐点卷积
    # stride=1 步距为一  padding=0 无填充   扩张卷积与原始输入相同

    # vertiacl方向上的分组卷积（逐通道卷积 depthwise convolution）
    depthwise_vertical_layer = torch.nn.Conv2d(in_channels=vertical.shape[1], 
            out_channels=vertical.shape[1], kernel_size=(vertical.shape[0], 1),
            stride=1, padding=(layer.padding[0], 0), dilation=layer.dilation,
            groups=vertical.shape[1], bias=False)
    # in_channels=vertical.shape[1]=R
    # out_channels=vertical.shape[1]=R
    # kernel_size=(vertical.shape[0], 1)=卷积核x方向上的大小
    # groups=vertical.shape[1]=R 设定逐通道卷积

    # horizontal方向上的分组卷积（逐通道卷积 depthwise convolution）
    depthwise_horizontal_layer = \
        torch.nn.Conv2d(in_channels=horizontal.shape[1],
            out_channels=horizontal.shape[1], 
            kernel_size=(1, horizontal.shape[0]), stride=layer.stride,
            padding=(0, layer.padding[0]), 
            dilation=layer.dilation, groups=horizontal.shape[1], bias=False)
    # in_channels=horizontal.shape[1]=R
    # out_channels=horizontal.shape[1]=R
    # kernel_size=(1, horizontal.shape[0])=卷积核y方向上的大小
    # groups=horizontal.shape[1]=R 设定逐通道卷积
    
    # R(CP分解秩) to T(目标输出通道数) 的通道压缩逐点卷积（pointwise convolution）
    pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
            out_channels=last.shape[0], kernel_size=1, stride=1,
            padding=0, dilation=layer.dilation, bias=True)
    # in_channels=last.shape[0]=R
    # out_channels=first.shape[1]=T
    # kernel_size=1 代表逐点卷积
    # stride=1 步距为一  padding=0 无填充   扩张卷积与原始输入相同

    pointwise_r_to_t_layer.bias.data = layer.bias.data  # 使用与原始层相同的偏置

    # depthwise_horizontal_layer 层的权重大小为 (R, 1, 1, X)
    # horizontal 的大小为 (X, R)
    depthwise_horizontal_layer.weight.data = \
        torch.transpose(horizontal, 1, 0).unsqueeze(1).unsqueeze(1)
        
    # depthwise_vertical_layer 层的权重大小为 (R, 1, Y, 1)
    # vertical 的大小为 (Y, R)
    depthwise_vertical_layer.weight.data = \
        torch.transpose(vertical, 1, 0).unsqueeze(1).unsqueeze(-1)
        
    # pointwise_s_to_r_layer 层的权重张量大小为 (R, S, 1, 1)
    # first 的大小为 (S, R)
    # 调整第零维与第一维，并在中间添加两个维度
    pointwise_s_to_r_layer.weight.data = \
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
        
    # pointwise_r_to_t_layer 层的权重张量大小为 (T, R, 1, 1)
    # last 的大小为 (T, R)
    pointwise_r_to_t_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)

    # 包装层
    new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer, \
                    depthwise_horizontal_layer, pointwise_r_to_t_layer]
    
    return nn.Sequential(*new_layers)


def tucker_decomposition_conv_layer(layer, ranks):
    
    """ 
        layer: torch.nnConv2d   pytorch二维卷积层
        ranks:  int              卷积层张量 Tucker 分解核心张量
        
        Gets a conv layer,
        给定一个卷积层
        returns a nn.Sequential object with the Tucker decomposition.
        返回一个经过 Tucker 分解的 nn.Sequential 对象
    """
    
    # 沿着输出通道（卷积核核数、输出特征图数）与输入通道（输入通道数）进行分解！！！！！
    core, [last, first] = partial_tucker(layer.weight.data, modes=[0, 1], rank=ranks, init='svd')   # 这里他娘的写错了，哪来的 ranks=ranks?
    # core  大小为 (rank[0], rank[1], layer.weight.data.shape[2], layer.weight.data[3])=(R4, R3, layer.weight.data.shape[2], layer.weight.data[3])
    # core  具有的 dim_2 与 dim_3 大小与原卷积核大小相同
    # last  大小为 (layer.weight.data.shape[0], rank[0])=(layer.weight.data.shape[0], R4)
    # first 大小为 (layer.weight.data.shape[1], rank[1])=(layer.weight.data.shape[1], R3)

    # A pointwise convolution that reduces the channels from S to R3
    # 使用逐点卷积将通道数由 S 减少为 R3
    first_layer = torch.nn.Conv2d(
        in_channels=first.shape[0], 
        out_channels=first.shape[1], 
        kernel_size=1,
        stride=1, 
        padding=0, 
        dilation=layer.dilation, 
        bias=False
    )
    # in_channels=first.shape[0]=S
    # out_channels=first.shape[1]=R3

    # A regular 2D convolution layer with R3 input channels and R4 output channels
    # 使用常规卷积（大小与原卷积核大小相同），输入通道数为 R3
    core_layer = torch.nn.Conv2d(
        in_channels=core.shape[1], 
        out_channels=core.shape[0], 
        kernel_size=layer.kernel_size,
        stride=layer.stride, 
        padding=layer.padding, 
        dilation=layer.dilation,
        bias=False
    )
    
    # in_channels=core.shape[1]=R3
    # out_channels=core.shape[0]=R4
    # kernel_size=layer.kernel_size=(core.shape[2], core.shape[3])

    # A pointwise convolution that increases the channels from R4 to T
    # 使用逐点卷积将通道数由 R4 增加为 T
    last_layer = torch.nn.Conv2d(
        in_channels=last.shape[1], 
        out_channels=last.shape[0], 
        kernel_size=1, 
        stride=1,
        padding=0, 
        dilation=layer.dilation, 
        bias=True
    )
    # in_channels=first.shape[0]=R4
    # out_channels=first.shape[1]=T

    last_layer.bias.data = layer.bias.data  # 加偏置

    # first_layer层的权重大小为 (R3, layer.weight.data.shape[1], 1, 1)
    # first 大小为 (layer.weight.data.shape[1], rank[1])=(layer.weight.data.shape[1], R3)
    # 调整两个维度，并在最后添加两个维度
    first_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    
    # last_layer层的权重大小为 (layer.weight.data.shape[0], R4, 1, 1)
    # last 大小为 (layer.weight.data.shape[0], rank[0])=(layer.weight.data.shape[0], R4)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    
    core_layer.weight.data = core
    
    new_layers = [first_layer, core_layer, last_layer]
    
    return nn.Sequential(*new_layers)


class Tucker_CNN(nn.Module):
    
    '''
        思想很简单，将输入通道先进行减小
        进行常规卷积后，将输出通道进行恢复
    '''
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, ranks: tuple, device='cuda:0', **kwargs):
        
        '''
            对输入与输出通道数进行降维
            in_channels:  int     输入通道数
            out_channels: int     输出通道数
            kernel_size:  int     模拟卷积核大小
            stride:       int     卷积步长
            padding:      int     zero填充数目
            ranks:        tuple   降维参数
            ranks[0]为降维输入通道数
            ranks[1]为降维输出通道数
            注意，使用该层后，就可以不采用 BN 层了
        '''
        
        super(Tucker_CNN, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ranks = ranks
        self.device = device
        
        # 输入通道降维
        self.first_layer = torch.nn.Conv2d(
            in_channels = self.in_channels,
            out_channels = self.ranks[0],
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = False,
            device = self.device
        )
        
        # 常规卷积
        self.core_layer = torch.nn.Conv2d(
            in_channels = self.ranks[0],
            out_channels = self.ranks[1],
            kernel_size = self.kernel_size,
            stride = 1,
            padding = self.padding,
            bias = False,
            device = self.device
        )
        
        # 通道恢复
        self.last_layer = torch.nn.Conv2d(
            in_channels = self.ranks[1],
            out_channels = self.out_channels,
            kernel_size = 1,
            stride = self.stride,
            padding = 0,
            bias = True,
            device = self.device
        )

    def forward(self, x):
        
        x = self.first_layer(x)
        x = nn.BatchNorm2d(self.ranks[0], affine=True, device=self.device)(x)
        x = self.core_layer(x)
        x = nn.BatchNorm2d(self.ranks[1], affine=True, device=self.device)(x)
        x = self.last_layer(x)
        x = nn.BatchNorm2d(self.out_channels, affine=True, device=self.device)(x)
        
        return x


class CP_CNN(nn.Module):
    
    '''
        思想很简单，将输入通道先进行减小
        将常规卷积转换为两个方向上的卷积，将输出通道进行恢复
    '''
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, rank: int, device='cuda:0', **kwargs):
        
        '''
            in_channels:  int     输入通道数
            out_channels: int     输出通道数
            kernel_size:  int     模拟卷积核大小
            stride:       int     卷积步长
            padding:      int     zero填充数目
            ranks:        int     降维参数
            注意，使用该层后，就可以不采用 BN 层了
        '''
        
        super(CP_CNN, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.rank = rank
        self.device = device
        
        # 输入通道降维
        self.pointwise_s_to_r_layer = torch.nn.Conv2d(
            in_channels = self.in_channels,
            out_channels = self.rank,
            kernel_size=1,
            stride = 1,
            padding = 0,
            bias = False,
            device = self.device
        )
        
        # vertical 方向逐层卷积
        self.depthwise_vertical_layer = torch.nn.Conv2d(
            in_channels = self.rank,
            out_channels = self.rank,
            kernel_size = (self.kernel_size, 1),
            stride = 1,
            padding = (self.padding, 0),
            groups = self.rank,
            bias = False ,
            device = self.device
        )
        
        # horizontal 方向逐层卷积
        self.depthwise_horizontal_layer = torch.nn.Conv2d(
            in_channels = self.rank,
            out_channels = self.rank,
            kernel_size = (1, self.kernel_size),
            stride = 1,
            padding = (0, self.padding),
            groups = self.rank,
            bias = False,
            device = self.device
        )
        
        # 输出通道恢复
        self.pointwise_r_to_t_layer = torch.nn.Conv2d(
            in_channels = self.rank,
            out_channels = self.out_channels,
            kernel_size = 1,
            stride = self.stride,
            padding = 0,
            bias = True,
            device = self.device
        )
        
    def forward(self, x):
        
        x = self.pointwise_s_to_r_layer(x)
        x = nn.BatchNorm2d(self.rank, affine=True, device=self.device)(x)
        x = self.depthwise_horizontal_layer(x)
        x = nn.BatchNorm2d(self.rank, affine=True, device=self.device)(x)
        x = self.depthwise_vertical_layer(x)
        x = nn.BatchNorm2d(self.rank, affine=True, device=self.device)(x)
        x = self.pointwise_r_to_t_layer(x)
        x = nn.BatchNorm2d(self.out_channels, affine=True, device=self.device)(x)
        
        return x