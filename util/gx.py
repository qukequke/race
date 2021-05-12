import matplotlib.pyplot as plt
import numpy as np

def fx(x, time_1, time_2):
    return (x**time_1) *((1-x)**time_2)

def Gaussian(x, u, d):
    """
    参数:
    x -- 变量
    u -- 均值
    d -- 标准差

    返回:
    p -- 高斯分布值
    """
    d_2 = d * d * 2
    zhishu = -(np.square(x - u) / d_2)
    exp = np.exp(zhishu)
    pi = np.pi
    xishu = 1 / (np.sqrt(2 * pi) * d)
    p = xishu * exp
    return p

def get_max(x, y):
    max_x_index = np.argmax(y)
    max_x = x[max_x_index]
    max_y = y[max_x_index]
    return max_x, max_y


x = np.linspace(0, 1, 200)
y1 = fx(x, 7, 3)
y2 = fx(x, 700, 300)
g = Gaussian(x, 0.5, 0.1)
y = g*y1
y_list = [y1, y1*g, y2*g]
for y in y_list:
    plt.figure()
    max_x, max_y = get_max(x, y)
    plt.plot(x, y)
    plt.text(max_x, max_y, f'{(max_x, max_y)}')
    plt.vlines(max_x, 0, max_y, colors='r', linestyles='dashed')
    plt.hlines(max_y, 0, max_x, colors='r', linestyles='dashed')
    plt.show()
    plt.figure()