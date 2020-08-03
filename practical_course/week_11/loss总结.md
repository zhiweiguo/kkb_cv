## torch.nn 与torch.nn.functional的区别于联系

```
import torch
import torch.nn as nn
import torch.nn.functional as F
```

区别：torch.nn中的模块可以保存参数的信息，而functional模块中需要在每次调用时传参；

联系：torch.nn模块中的功能都是调用functional模块来实现，也就是说两者功能保持一致，只是torch.nn是在functional的基础上在外面包了一层，用于存储参数信息。

## 1. softmax

softmax的数学含义如下图所示：

![](F:\总结\kkb_cv\practical_course\week_11\softmax.jpg)

![](F:\总结\kkb_cv\practical_course\week_11\softmax2.jpg)

简言之，softmax的操作可以放大数组元素之间的差异，且同时保持所有经过softmax之后的数组，所有元素值的和为1，因为可以通过softmax的值来代表概率，用于分类网络中计算交叉熵损失。

### nn.Softmax()

![](F:\总结\kkb_cv\practical_course\week_11\softmax3.jpg)

使用示例：

```
m = nn.Softmax(dim=2)        # 可以先定义出接口，其中dim参数可选，默认是1
input = torch.rand((3,4,5,6))
output = m(input)
print(torch.sum(output, dim=2))    # 此时输出全为1
```

### F.softmax()

与nn.Softmax的功能一样，使用方法如下：

```
input = torch.rand((3,4,5,6))
output = F.softmax(input, dim=2) # 直接调用，且设置dim，默认为1
print(torch.sum(output, dim=2))  # 此时输出全为1
```

## 2. Log softmax

log softmax的数学含义是：在softmax的基础上再进行一次log操作，如下所示：

log_softmax(input) = log(softmax(input))

因为softmax之后的概率值都是0~1范围内，所以经过log操作后，所有的数值都变为负值。

### nn.LogSoftmax()

使用示例：

```
m = nn.LogSoftmax(dim=2)        # 可以先定义出接口，其中dim参数可选，默认是1
input = torch.rand((3,4,5,6))
output = m(input)    
```

### F.log_softmax()

使用示例：

```
input = torch.rand((3,4,5,6))
output = F.log_softmax(input, dim=2) # 直接调用，且设置dim，默认为1
# 验证
output2 = torch.log(F.softmax(input, dim=2))
print(output.equal(output2))    # 结果为True，说明两者结果一致
```

## 3. NLL loss

nll loss是指 log likelihood loss，即负对数似然损失。

输入是包含类别log probabilities的数据，因此一般需要在网络的最后一层增加一个求log的操作层，常见的是使用log_softmax

损失的计算是可以看如下描述：

![](F:\总结\kkb_cv\practical_course\week_11\nllloss.jpg)

### nn.NLLLoss()

![](F:\总结\kkb_cv\practical_course\week_11\nllloss2.jpg)

使用示例：

```
m = nn.LogSoftmax()      # 定义logsoftmax
loss = nn.NLLLoss()      # 定义nll loss
# input is of size nBatch x nClasses = 3 x 5
input = autograd.Variable(torch.randn(3, 5), requires_grad=True) # 创建input
# each element in target has to have 0 <= value < nclasses
target = autograd.Variable(torch.LongTensor([1, 0, 4]))  # 标签
output = loss(m(input), target)      # 计算loss
output.backward()   # 反向传播
```

### F.nll_loss()

使用示例：

```
# input is of size nBatch x nClasses = 3 x 5
input = autograd.Variable(torch.randn(3, 5), requires_grad=True) # 创建input
# each element in target has to have 0 <= value < nclasses
target = autograd.Variable(torch.LongTensor([1, 0, 4]))  # 标签
output = F.nll_loss(F.log_softmax(input), target)      # 计算loss
output.backward()   # 反向传播
```

## cross entropy loss

交叉熵损失用于分类网络中，在pytorch中的交叉熵损失相当于logsoftmax与nllloss的结合。

![](F:\总结\kkb_cv\practical_course\week_11\celoss.jpg)

### nn.CrossEntropyLoss()

输入参数的设置与Nll loss的设置一样。

使用示例:

```
# input is of size N x C = 3 x 5
input = torch.randn(3, 5, requires_grad=True)     # 输入input
# each element in target has to have 0 <= value < C
target = torch.tensor([1, 0, 4])         # 标签
loss = nn.CrossEntropyLoss(dim=1) # 定义ce loss
output = loss(input, target) # 计算 loss
print(output)
```

### F.cross_entropy()

使用示例：

```
input = torch.randn(3, 5, requires_grad=True)         # 输入input
# each element in target has to have 0 <= value < C
target = torch.tensor([1, 0, 4])
loss = F.cross_entropy(input, target)
loss.backward()
```

## BCE loss

计算 `target` 与 `output` 之间的二进制交叉熵

当不指定weights时：
$$
loss(o,t)=-\frac{1}{n}\sum_i(t[i] log(o[i])+(1-t[i]) log(1-o[i]))
$$
当指定weights时：
$$
loss(o,t)=-\frac{1}{n}\sum_iweights[i] (t[i] log(o[i])+(1-t[i])* log(1-o[i])) 
$$
默认情况下，loss会基于`element`平均，如果`size_average=False`的话，`loss`会被累加。

### nn.BCELoss()

使用示例：

```
m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
output = loss(m(input), target)
output.backward()
```

### F.bce_loss()

使用示例：

```
input = torch.randn((3, 2), requires_grad=True)
target = torch.rand((3, 2), requires_grad=False)
loss = F.binary_cross_entropy(F.sigmoid(input), target)
loss.backward()
```

## 参考链接

1. https://blog.csdn.net/qq_22210253/article/details/85229988
2. https://www.jianshu.com/p/35060b7553c8
3. https://blog.csdn.net/geter_CS/article/details/84857220
4. https://blog.csdn.net/hao5335156/article/details/80607732
5. https://www.cnblogs.com/wanghui-garcia/p/10862733.html
6. https://blog.csdn.net/shanglianlm/article/details/85019768