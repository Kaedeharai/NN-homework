import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import Lambda
from tqdm import trange


train_data = datasets.MNIST("data", train=True, download=True,
                            target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
test_data = datasets.MNIST("data", train=False, download=True, 
                            target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

print(len(train_data))
print(len(test_data))

tmp1=[]
tmp2=[]
for i in range(len(train_data)):
    tmp1.append(np.array(train_data[i][0]))
    tmp2.append(np.array(train_data[i][1]))
x_train = np.array(tmp1)
y_train = np.array(tmp2)

x_train = x_train.reshape(x_train.shape[0], 28 * 28)


tmp1=[]
tmp2=[]
for i in range(len(test_data)):
    tmp1.append(np.array(test_data[i][0]))
    tmp2.append(np.array(test_data[i][1]))    
x_test = np.array(tmp1)
y_test = np.array(tmp2)

x_test = x_test.reshape(x_test.shape[0], 28 * 28)


class Layer:
    def __init__(self):
        self.mem = {}
        
    def forward(self, x, W):
        # H = x * W + b
        H = np.matmul(x, W)

        self.mem = {'x': x, 'W': W}
        return H

    def backward(self, grad_H):
        '''
        param {
        grad_H: 损失函数关于 H 的梯度
        }
        '''
        x = self.mem['x']
        W = self.mem['W']
        
        '''
        偏导数 : 损失函数关于 H 的梯度 * H 关于 w 的梯度
        '''
        grad_X = np.matmul(grad_H, W.T)
        grad_W = np.matmul(x.T, grad_H)
        
        return grad_X, grad_W

class Relu:
    def __init__(self):
        self.mem = {}
        
    def forward(self, x):
        self.mem['x'] = x
        return np.where(x > 0, x, np.zeros_like(x))
    
    def backward(self, grad_f):
        '''
        param {
        grad_f: 损失函数关于 f 的梯度
        }
        return {
        grad_H: 损失函数关于 H 的梯度
        }
        '''
        x = self.mem['x']
        grad_H = (x > 0).astype(np.float32) * grad_f
        return grad_H


class Softmax():
    def __init__(self):
        self.mem = {}
        self.epsilon = 1e-12
 
    def forward(self, x):
        c = np.max(x)
        x_exp = np.exp(x - c)
        denominator = np.sum(x_exp, axis=1, keepdims=True)
        out = x_exp/(denominator + self.epsilon)
        self.mem["out"] = out
        self.mem["x_exp"] = x_exp
        return out
 
    def backward(self, grad_y):
        s = self.mem["out"]
        sisj = np.matmul(np.expand_dims(s, axis=2), np.expand_dims(s,axis=1))
        g_y_exp = np.expand_dims(grad_y, axis=1)
        tmp = np.matmul(g_y_exp, sisj)
        tmp = np.squeeze(tmp, axis=1)
        softmax_grad = -tmp + grad_y*s
        return softmax_grad


class CrossEntropy():
    def __init__(self):
        self.mem = {}
        self.epsilon = 1e-12
    def forward(self, x, labels):
        log_prob = np.log(x + self.epsilon)
        out = np.mean(np.sum(-log_prob*labels,axis=1))
        self.mem["x"] = x
        return out
    def backward(self, labels):
        x = self.mem["x"]
        return -1/(x + self.epsilon)*labels

class myModel:
    def __init__(self):
        self.W1 = np.random.normal(loc = 0, scale = 1, size = [28 * 28 + 1, 512]) / np.sqrt(784 / 2)
        self.W2 = np.random.normal(loc = 0, scale = 1, size = [512, 256]) / np.sqrt(512 / 2)
        self.W3 = np.random.normal(loc = 0, scale = 1, size = [256, 10]) / np.sqrt(256 / 2)

        self.mul_h1 = Layer()
        self.relu_1 = Relu()
        self.mul_h2 = Layer()
        self.relu_2 = Relu()
        self.mul_h3 = Layer()
        self.softmax = Softmax()
        self.cross_en = CrossEntropy()

    def forward(self, x, labels):
        bias = np.ones(shape=[x.shape[0], 1])
        x = np.concatenate([x, bias], axis=1)
        self.h1 =self.mul_h1.forward(x, self.W1)
        self.h1_relu = self.relu_1.forward(self.h1)
        self.h2 = self.mul_h2.forward(self.h1_relu, self.W2)
        self.h2_relu = self.relu_2.forward(self.h2)
        self.h3 = self.mul_h3.forward(self.h2_relu, self.W3)
        self.h3_soft = self.softmax.forward(self.h3)
        self.loss = self.cross_en.forward(self.h3_soft, labels)
 
    def backward(self, labels):
        self.loss_grad = self.cross_en.backward(labels)
        self.h3_soft_grad = self.softmax.backward(self.loss_grad)
        self.h3_grad, self.W3_grad = self.mul_h3.backward(self.h3_soft_grad)
        self.h2_relu_grad = self.relu_2.backward(self.h3_grad)
        self.h2_grad, self.W2_grad = self.mul_h2.backward(self.h2_relu_grad)
        self.h1_relu_grad = self.relu_1.backward(self.h2_grad)
        self.h1_grad, self.W1_grad = self.mul_h1.backward(self.h1_relu_grad)

# 计算精确度
def computeAccuracy(prob, labels):
    predicitions = np.argmax(prob, axis=1)
    truth = np.argmax(labels, axis=1)
    return np.mean(predicitions == truth)

def trainOneStep(model, x, y, lr = 1e-5):
    model.forward(x, y)
    model.backward(y)
    model.W1 -= lr * model.W1_grad
    model.W2 -= lr* model.W2_grad
    model.W3 -= lr * model.W3_grad
    loss = model.loss
    accuracy = computeAccuracy(model.h3_soft, y)
    return loss, accuracy

# 评估模型
def evaluate(model, x, y):
    model.forward(x, y)
    loss = model.loss
    accuracy = computeAccuracy(model.h3_soft, y)
    return loss, accuracy

model = myModel()

 
for epoch in range(200):
    loss, accuracy = trainOneStep(model, x_train, y_train, 0.001)
    print('epoch',epoch,'loss',loss,'accuracy',accuracy)


loss, accuracy = evaluate(model, x_test, y_test)
print(f'Evaluate the best model, test loss={loss:0<10.8}, accuracy={accuracy:0<8.6}.')