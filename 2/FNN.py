import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import Lambda
from tqdm.std import trange


class FullConnectionLayer():
    def __init__(self):
        self.mem = {}

    def forward(self, X, W):
        '''
        param {
            X: shape(m, d), 前向传播输入矩阵
            W: shape(d, d'), 前向传播权重矩阵
        }
        return {
            H: shape(m, d'), 前向传播输出矩阵
        }
        '''
        self.mem = {'X': X, 'W': W}
        H = np.matmul(X, W)
        return H

    def backward(self, grad_H):
        '''
        param {
            grad_H: shape(m, d'), Loss 关于 H 的梯度
        }
        return {
            grad_X: shape(m, d), Loss 关于 X 的梯度
            grad_W: shape(d, d'), Loss 关于 W 的梯度
        }
        '''
        X = self.mem['X']
        W = self.mem['W']
        grad_X = np.matmul(grad_H, W.T)
        grad_W = np.matmul(X.T, grad_H)
        return grad_X, grad_W

class Relu():
    def __init__(self):
        self.mem = {}

    def forward(self, X):
        self.mem["X"] = X
        return np.where(X > 0, X, np.zeros_like(X))

    def backward(self, grad_y):
        X = self.mem["X"]
        return (X > 0).astype(np.float32) * grad_y

class Softmax():
    def __init__(self):
        self.mem = {}
        self.epsilon = 1e-12

    def forward(self, p):
        c = np.max(p)
        p_exp = np.exp(p - c)
        denominator = np.sum(p_exp, axis=1, keepdims=True)
        s = p_exp / (denominator + self.epsilon)
        self.mem["s"] = s
        self.mem["p_exp"] = p_exp
        return s

    def backward(self, grad_s):
        s = self.mem["s"]
        sisj = np.matmul(np.expand_dims(s, axis=2), np.expand_dims(s, axis=1))
        tmp = np.matmul(np.expand_dims(grad_s, axis=1), sisj)
        tmp = np.squeeze(tmp, axis=1)
        grad_p = -tmp + grad_s * s
        return grad_p

class CrossEntropy():
    def __init__(self):
        self.mem = {}
        self.epsilon = 1e-12

    def forward(self, p, y):
        self.mem['p'] = p
        log_p = np.log(p + self.epsilon)
        return np.mean(np.sum(-y * log_p, axis=1))

    def backward(self, y):
        p = self.mem['p']
        return -y * (1 / (p + self.epsilon))

class FullConnectionModel():
    def __init__(self, latent_dims):
        self.W1 = np.random.normal(loc=0, scale=1, size=[28 * 28 + 1, latent_dims]) / np.sqrt((28 * 28 + 1) / 2)
        self.W2 = np.random.normal(loc=0, scale=1, size=[latent_dims, 10]) / np.sqrt(latent_dims / 2)

        
        self.mul_h1 = FullConnectionLayer()
        self.relu = Relu()
        self.mul_h2 = FullConnectionLayer()
        self.softmax = Softmax()
        self.cross_en = CrossEntropy()

    def forward(self, x, label):
        bias = np.ones(shape = [x.shape[0], 1])
        x = np.concatenate([x, bias], axis = 1)
        self.h1 = self.mul_h1.forward(x, self.W1)
        self.h1_relu = self.relu.forward(self.h1)
        self.h2 = self.mul_h2.forward(self.h1_relu, self.W2)
        self.h2_soft = self.softmax.forward(self.h2)
        self.loss = self.cross_en.forward(self.h2_soft, label)

    def backward(self, label):
        self.loss_grad = self.cross_en.backward(label)
        self.h2_soft_grad = self.softmax.backward(self.loss_grad)
        self.h2_grad, self.W2_grad = self.mul_h2.backward(self.h2_soft_grad)
        self.h1_relu_grad = self.relu.backward(self.h2_grad)
        self.h1_grad, self.W1_grad = self.mul_h1.backward(self.h1_relu_grad)

def computeAccuracy(prob, labels):
    predicitions = np.argmax(prob, axis=1)
    truth = np.argmax(labels, axis=1)
    return np.mean(predicitions == truth)

def trainOneStep(model, x_train, y_train, learning_rate=1e-5):
    model.forward(x_train, y_train)
    model.backward(y_train)
    model.W1 -= learning_rate * model.W1_grad
    model.W2 -= learning_rate * model.W2_grad
    loss = model.loss
    accuracy = computeAccuracy(model.h2_soft, y_train)
    return loss, accuracy

def train(x_train, y_train, x_validation, y_validation):
    epochs = 200
    learning_rate = 0.0005
    latent_dims_list = [100, 200, 300]
    best_accuracy = 0
    best_latent_dims = 0

    # 验证集
    print("Start seaching the best parameter...\n")
    for latent_dims in latent_dims_list:
        model = FullConnectionModel(latent_dims)

        bar = trange(20)
        for epoch in bar:
            loss, accuracy = trainOneStep(model, x_train, y_train, learning_rate) 
            bar.set_description(f'Parameter latent_dims={latent_dims: <3}, epoch={epoch + 1: <3}, loss={loss: <10.8}, accuracy={accuracy: <8.6}')
        bar.close()

        validation_loss, validation_accuracy = evaluate(model, x_validation, y_validation)
        print(f"Parameter latent_dims={latent_dims: <3}, validation_loss={validation_loss}, validation_accuracy={validation_accuracy}.\n")

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_latent_dims = latent_dims

    print(f"The best parameter is {best_latent_dims}.\n")
    print("Start training the best model...")

    best_model = FullConnectionModel(392)
    x = np.concatenate([x_train, x_validation], axis=0)
    y = np.concatenate([y_train, y_validation], axis=0)
    bar = trange(epochs)
    for epoch in bar:
        loss, accuracy = trainOneStep(best_model, x, y, learning_rate)
        bar.set_description(f'Training the best model, epoch={epoch + 1: <3}, loss={loss: <10.8}, accuracy={accuracy: <8.6}')  # 给进度条加个描述
    bar.close()

    return best_model

# 评估模型
def evaluate(model, x, y):
    model.forward(x, y)
    loss = model.loss
    accuracy = computeAccuracy(model.h2_soft, y)
    return loss, accuracy


train_data = datasets.MNIST("data", train=True, download=True,
                            target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
test_data = datasets.MNIST("data", train=False, download=True, 
                            target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

tmp1=[]
tmp2=[]
for i in range(len(train_data)):
    tmp1.append(np.array(train_data[i][0]))
    tmp2.append(np.array(train_data[i][1]))
x_train = np.array(tmp1)
y_train = np.array(tmp2)

x_validation = x_train[48000:]
x_train = x_train[:48000]
y_validation = y_train[48000:]
y_train = y_train[:48000]

x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_validation = x_validation.reshape(x_validation.shape[0], 28 * 28)

tmp1=[]
tmp2=[]
for i in range(len(test_data)):
    tmp1.append(np.array(test_data[i][0]))
    tmp2.append(np.array(test_data[i][1]))    
x_test = np.array(tmp1)
y_test = np.array(tmp2)

x_test = x_test.reshape(x_test.shape[0], 28 * 28)


model = train(x_train, y_train, x_validation, y_validation)
loss, accuracy = evaluate(model, x_test, y_test)
print(f'Evaluate the best model, test loss={loss:0<10.8}, accuracy={accuracy:0<8.6}.')
