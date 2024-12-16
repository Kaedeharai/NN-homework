import numpy as np

"""

这个文件实现了各种一阶更新规则，这些规则通常用于训练神经网络。
每个更新规则接受当前权重和损失相对于这些权重的梯度，并生成下一组权重。
每个更新规则都有相同的接口:

def update(w, dw, config=None):

Inputs:
  - w: 给出当前权重的numpy数组。
  - dw: 一个与w形状相同的numpy数组，给出了
关于w的损失。
  - config: 包含超参数值(如learning)的字典速度、动量等。如果更新规则要求缓存多个值迭代，那么配置也将保存这些缓存值。

Returns:
  - next_w: 下一个点更新后。
  - config: 下一个迭代中要传递的配置字典更新规则。

NOTE: 对于大多数更新规则，默认的学习速率可能不会很好地执行;但是，其他超参数的默认值应该可以很好地用于各种不同的问题。

For efficiency, update rules may perform in-place updates, mutating w and setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: 与w和dw形状相同的numpy数组，用于存储梯度的移动平均。 
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: 实现动量更新公式。 将更新的值存储在next_w变量中。 你还应该使用和更新速度v。      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    使用RMSProp更新规则，该规则使用梯度值平方的移动平均值来设置自适应的每个参数学习率。

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: 梯度二阶矩的移动平均。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: 实现RMSprop更新公式，将w的下一个值存储在next_w变量中。                                                              #     不要忘记更新配置['cache']中存储的缓存值。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    使用adam更新规则，其中包括梯度和其平方的移动平均和偏差修正项。 

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: 梯度移动平均。
    - v: 梯度平方的移动平均。
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    ###########################################################################
    # TODO: 实现Adam更新公式，将w的下一个值存储在next_w变量中。                                                         #
    #   不要忘记更新配置中存储的m、v和t变量。                                                                      #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations. 为了匹配参考输出，请在使用它进行任何计算之前修改t。                                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config
