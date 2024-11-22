import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos

R = 1e6
C = 1e-6
a = 1 / (R * C)

t_min = -1.0
t_max = 15.0
n = 1000
tau = (t_max - t_min) / n

omega = [pi, 1 / 2 * pi, 1 / 5 * pi]
color = ['blue', 'green', 'red']


def u(time):
    if time < 0:
        return 0
    else:
        return 1


def x_f(t, omega_0):
    return cos(omega_0 * t) * u(t)


def a_9_c_1():
    t = np.linspace(t_min, t_max, n)

    x_real = np.zeros((len(omega), t.size))
    y_real = np.zeros((len(omega), t.size))

    for i in range(len(omega)):
        x_real[i] = np.vectorize(x_f)(t, omega[i])
        for j in range(1,n):
            y_real[i][j] = 1 / (1 + a * tau) * (y_real[i][j - 1] + x_real[i][j] - x_real[i][j - 1])

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    for i in range(len(omega)):
        plt.plot(t, x_real[i], color=color[i])
    plt.ylabel('$x_R(t)$')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    for i in range(len(omega)):
        plt.plot(t, y_real[i], color=color[i])
    plt.ylabel('$y_R(t)$')
    plt.xlabel('$t$')
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    a_9_c_1()
