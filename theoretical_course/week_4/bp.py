import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data

class BP(nn.Module):
    """
    包含2个隐藏层的BP神经网络
    input_num: 模型输入的维度
    output_num: 模型输出的维度
    hidden_num_list: 隐藏层的维度
    use_bias: 是否使用激活函数
    act: 激活函数的名称
    """
    def __init__(self, input_num, output_num, hidden_num_list=[10, 15], use_bias=True, act='sigmoid'):
        super(BP, self).__init__()
        self.use_bias = use_bias
        # 通过nn.Parameter把tensor添加到module.parameters()中，便于后续直接自动更新梯度
        self.w1 = nn.Parameter(torch.rand(input_num, hidden_num_list[0], requires_grad=True))
        self.b1 = nn.Parameter(torch.rand(1, hidden_num_list[0], requires_grad=True))
        self.w2 = nn.Parameter(torch.rand(hidden_num_list[0], hidden_num_list[1], requires_grad=True))
        self.b2 = nn.Parameter(torch.rand(1, hidden_num_list[1], requires_grad=True))
        self.w3 = nn.Parameter(torch.rand(hidden_num_list[1], output_num, requires_grad=True))
        self.act = self.get_act(act)

    def get_act(self, act):
        """
        激活函数
        """
        if act == 'sigmoid':
            act_fun = nn.Sigmoid()
        elif act == 'tanh':
            act_fun = nn.Tanh()
        elif act == 'relu':
            act_fun = nn.ReLU()
        else:
            act_fun = None
        return act_fun  
        
    def forward(self, x):        
        #x = torch.tensor(x, dtype=torch.float32)
        x = x.mm(self.w1)
        x = x.add(self.b1)
        x = self.act(x)
        x = x.mm(self.w2)
        x = x.add(self.b2)
        x = self.act(x)
        x = x.mm(self.w3)
        return x


if __name__ == "__main__":
    epoch = 100
    input_dim = 25
    batch_size = 512
    output_dim = 10
    hidden_dim = [60,40]
    class_num = 10
    data_set = data.MnistDataSet('./mnist',feature_dim=input_dim, batch_size=batch_size)
    bp_net = BP(input_dim, output_dim, hidden_dim, True, 'sigmoid')

    optimizer = optim.SGD(bp_net.parameters(), lr=0.01, momentum=0.9)
    #criterion = nn.MSELoss(reduction='mean')
    criterion = nn.CrossEntropyLoss()

    for i in range(epoch):
        for batch_data, batch_label in data_set.next():
            input = torch.from_numpy(batch_data).float()
            #label = torch.from_numpy(batch_label).view([len(batch_label),1]).long()
            label = torch.from_numpy(batch_label).long()
            #onehot_label = F.one_hot(label, class_num)
            #onehot_label = torch.squeeze(onehot_label)
            output = bp_net(input)
            #output_softmax = torch.softmax(output)
            #print(output.shape)
            loss = criterion(output, label) 
            print("epoch={},loss={}".format(i,loss))
            optimizer.zero_grad()
            #bp_net.zero_grad()
            loss.backward()
            optimizer.step()
            #bp_net.grad_update(0.001)
    
    # 验证准确率
    bp_net.eval()
    test_num = len(data_set.test_labels)
    test_input = torch.from_numpy(data_set.test_features).float()
    test_labels = torch.from_numpy(data_set.test_labels).int()
    # 预测
    test_output = bp_net(test_input)
    test_output = torch.squeeze(test_output)    # test_output = torch.argmax(test_output, dim=1)
    test_output = torch.round(test_output)
    equal_arr = torch.eq(test_output, test_labels)
    right_count = torch.sum(equal_arr)
    acc = right_count.numpy() / test_num * 100
    print("测试样本总数:{},预测正确样本数:{},模型准确率:{}%".format(test_num, right_count, acc))



            