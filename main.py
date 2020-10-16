import random
import numpy as np
import torch as t
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd
import os


def sigmoid(x):
    """
    :param x:  channel x batch
    :return: sigmoid
    """
    return t.ones_like(x) / (t.ones_like(x) + t.exp(-x))


def dsigmoid(x):
    return sigmoid(x) * (t.ones_like(x) - sigmoid(x))


def ReLU(x):
    x = t.where(x > 0, x, t.zeros_like(x))
    return x


def dReLU(x):
    x = t.where(x > 0, t.ones_like(x), t.zeros_like(x))
    return t.zeros_like(x)


def softmax(x):
    """
    :param x: channel x batch
    :return:
    """
    x = t.exp(x)
    x = x / t.sum(x, dim=0, keepdim=True)

    return x


class iris_dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = t.FloatTensor(self.data[index][:4])
        if self.data[index][4] == 'Iris-setosa':
            label = 0
        elif self.data[index][4] == 'Iris-versicolor':
            label = 1
        elif self.data[index][4] == 'Iris-virginica':
            label = 2

        return x, int(label)

    def __len__(self):
        return len(self.data)


class DNN(object):
    def __init__(self, layers, activation='sigmoid', learning_rate=0.01, requires_grad=False, cuda=False):
        self.layers = layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.cuda = cuda

        self.caches = {}
        self.grads = {}

        self.parameters = {}

        for i in range(1, len(self.layers)):
            # 满足标准正态分布初始化参数
            if self.cuda is False:
                self.parameters["w"+str(i)] = t.randn(self.layers[i], self.layers[i-1],
                                                      requires_grad=requires_grad)
            else:
                self.parameters["w"+str(i)] = t.randn(self.layers[i], self.layers[i-1],
                                                      requires_grad=requires_grad).cuda()

    def forward(self, x):
        """
        x: input: Batch * Channel
        a: output of the  nerve cell
        z: input of the nerve cell
        """
        if self.cuda is False:
            x = x.t()
        else:
            x = x.t().cuda()

        a = []
        z = []
        a.append(x)
        z.append(x)

        len_layers = len(self.parameters)

        for i in range(1, int(len_layers)):
            z.append(t.matmul(self.parameters["w"+str(i)], a[i-1]))

            if self.activation == 'sigmoid':
                a.append(sigmoid(z[-1]))
            elif self.activation == 'ReLU':
                a.append(ReLU(z[-1]))

        # output layer's activation is softmax
        z.append(t.matmul(self.parameters["w"+str(int(len_layers))], a[-1]))

        a.append(softmax(z[-1]))

        self.caches['a'] = a
        self.caches['z'] = z

        return a[-1].t()

    def backward(self, y):
        a = self.caches['a']
        z1 = self.caches['z'][-1]
        b = y.size(0)

        x = t.arange(0, b).long().view(-1, 1)
        y = y.long().view(-1, 1)
        c = t.cat([y, x], dim=1)

        len_layers = len(self.parameters)

        # calculate the gradient
        # z_ = z1.clone()

        z1[c[:, 0], c[:, 1]] -= 1

        # print(z_, '\n', z1, '\n', y)

        self.grads["dz"+str(int(len_layers))] = z1

        self.grads["dw"+str(int(len_layers))] = t.matmul(self.grads["dz"+str(int(len_layers))], a[-2].t()) / b

        for i in reversed(range(1, int(len_layers))):

            self.grads["dz"+str(i)] = t.matmul(self.parameters["w"+str(i+1)].t(), self.grads["dz"+str(i+1)])

            if self.activation == 'sigmoid':
                self.grads["dz" + str(i)] = t.mul(self.grads["dz"+str(i)], dsigmoid(self.caches['z'][i]))
            elif self.activation == 'ReLU':
                self.grads["dz" + str(i)] = t.mul(self.grads["dz"+str(i)], dReLU(self.caches['z'][i]))

            self.grads["dw"+str(i)] = t.matmul(self.grads["dz"+str(i)], a[i-1].t()) / b

        # update the parameters
        # for i in range(1, int(len_layers)):
        #     self.parameters["w"+str(i)] -= self.learning_rate * self.grads["dw"+str(i)]

    def compute_loss(self, y):
        # z1: batch x channel
        z1 = self.caches['z'][-1].t()

        if self.cuda is False:
            loss = F.cross_entropy(z1, y)
        else:
            loss = F.cross_entropy(z1, y.cuda())

        return loss


def prepare_dataset(root='/home/chenruoxi', path='Homework/BP/'):
    path_to_dataset = os.path.join(root, path + 'iris.data')
    data = pd.read_csv(open(path_to_dataset, 'r'))
    data = data.values.tolist()
    random.shuffle(data)

    return data


def accuracy(output, label):
    """
    :param output: batch * classes
    :param label: batch
    :return:
    """
    output = output.cpu()
    b = label.size(0)

    _, pred = output.topk(1, 1, True, True)

    pred = pred.t()
    correct = pred.eq(label.view(1, -1).long())

    correct = correct[0].view(-1).float().sum(0)
    res = correct.mul_(100.0 / b)

    return res


def train(model, train_loader):
    losses = []
    train_iterator = iter(train_loader)

    # for i in range(len(train_iterator)):
    for i in range(1):

        for k in range(0, len(model.parameters)):
            if model.parameters["w" + str(k+1)].grad is not None:
                model.parameters["w" + str(k+1)].grad.data.zero_()

        x, label = next(train_iterator)

        _ = model.forward(x)
        model.backward(label)

        loss = model.compute_loss(label)
        loss.backward()

        for k in range(0, len(model.parameters)):
            print("\nlayer {:03d}".format(k+1))
            g1 = model.parameters["w"+str(k+1)].grad
            g2 = model.grads["dw"+str(k+1)]

            print("autograd:{},{}".format(g1, g1.size()))
            print("manual:{}, {}".format(g2, g2.size()))

            print("{}".format((t.abs(g1 - g2) < t.ones_like(g1 - g2)*1e-10).all()))

        losses.append(loss.data.cpu().numpy())

    losses = np.array(losses)

    return np.mean(losses)


def test(model, test_loader):
    loss = []
    acc = []
    test_iterator = iter(test_loader)
    for i in range(len(test_iterator)):
        x, label = next(test_iterator)

        a1 = model.forward(x)

        loss.append(model.compute_loss(label))
        acc.append(accuracy(a1, label))

    loss = np.array(loss)
    acc = np.array(acc)

    return np.mean(loss), np.mean(acc)


if __name__ == '__main__':
    max_interation = 1
    batch_size = 1
    num_workers = 2
    freq_print = 50
    seed = 2020
    random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)

    model = DNN([4, 20, 20, 3], learning_rate=0.001, requires_grad=True, cuda=False)

    data = prepare_dataset()
    len_data = len(data)

    train_loader = DataLoader(iris_dataset(data[:int(len_data * 0.8)]), batch_size=batch_size, shuffle=True,
                              num_workers=1, pin_memory=True, drop_last=True)

    valid_loader = DataLoader(iris_dataset(data[int(len_data * 0.8):int(len_data * 0.9)]), batch_size=1, shuffle=True,
                              num_workers=1, pin_memory=True, drop_last=False)

    test_loader = DataLoader(iris_dataset(data[int(len_data * 0.9):]), batch_size=1, shuffle=True,
                             num_workers=1, pin_memory=True, drop_last=False)
    train_accuracies = []
    valid_accuracies = []
    train_losses = []
    valid_losses = []

    for epoch in range(max_interation):
        # train
        train_loss = train(model, train_loader)

        # accuracy on the train dataset
        # _, train_accuracy = test(model, train_loader)

        # loss and accuracy on the valid dataset
        # valid_loss, valid_accuracy = test(model, valid_loader)

        # update
        # train_losses.append(train_loss)
        # valid_losses.append(valid_loss)
        # train_accuracies.append(train_accuracy)
        # valid_accuracies.append(valid_accuracy)

        # print
        # if epoch % freq_print == 0 or epoch + 1 == max_interation:
        #     print("epoch:{:03d},train loss:{:.3f},train accuracy:{:.3f}, valid loss:{:.3f}, valid accuracy:{:.3f}"
        #           .format(epoch + 1, train_loss, train_accuracy, valid_loss, valid_accuracy))

    # test_loss, test_accuracy = test(model, test_loader)

    # print("finished training!\ntest loss:{:.3f}, test accuracy:{:.3f}".format(test_loss, test_accuracy))

