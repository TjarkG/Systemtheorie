import numpy as np
import matplotlib.pyplot as plt
from math import exp
from scipy import signal

from Übungsblatt_2_A9_b import impulse_response

R = 1e6
C = 1e-6
a = 1 / (R * C)

t_min = -10.0
t_max = 10.0
n = 10000
tau = (t_max - t_min) / n

sigma = [-1, -0.5, 0, 0.1, 0.5]
color = ['blue', 'green', 'red', 'black', 'purple']


def u(time):
    if time < 0:
        return 0
    else:
        return 1


def x_f(t, omega_0):
    return exp(omega_0 * t) * u(t)


def a_9_c_2():
    t = np.linspace(t_min, t_max, n)
    t_2 = np.linspace(t_min/tau, t_max/tau, n)

    x_real = np.zeros((len(sigma), t.size))
    y_real = np.zeros((len(sigma), t.size))
    h = np.vectorize(impulse_response)(t_2)

    for i in range(len(sigma)):
        x_real[i] = np.vectorize(x_f)(t, sigma[i])
        y_real[i] = signal.convolve(x_real[i], h, 'same')

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    for i in range(len(sigma)):
        plt.plot(t, x_real[i], color=color[i])
    plt.ylabel('$x_R(t)$')
    plt.ylim(-0.5, 3)
    plt.xlim(-1, 10)
    plt.grid(True)

    plt.subplot(2, 1, 2)
    for i in range(len(sigma)):
        plt.plot(t, y_real[i], color=color[i])
    plt.ylabel('$y_R(t)$')
    plt.xlabel('$t$')
    plt.ylim(-0.5, 1.5)
    plt.xlim(-1, 10)
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    a_9_c_2()
