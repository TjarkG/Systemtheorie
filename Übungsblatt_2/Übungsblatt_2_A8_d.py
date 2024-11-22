import numpy as np
import matplotlib.pyplot as plt
from cmath import exp, pi

R = 1
C = 1
a = 1 / (R*C)

omega = [1j * pi, 1j / 2 * pi, 1j / 5 * pi]
color = ['blue', 'green', 'red']


def u(time):
    if time < 0:
        return 0
    else:
        return 1


def transfer_function(s):
    return 1 / (C * (s + 1 / (R * C)))


def x_f(t, s):
    return exp(s * t) * u(t)


def y_f(t, s):
    return 1 / C * 1 / (s + 1 / (R * C)) * (np.exp(s * t) - np.exp(-1 / (R * C) * t)) * u(t)


def a_8_d():
    t = np.linspace(-1.0, 15.0, 1000)

    x_real = np.zeros((len(omega), t.size))
    x_imag = np.zeros((len(omega), t.size))
    y_real = np.zeros((len(omega), t.size))
    y_imag = np.zeros((len(omega), t.size))

    for i in range(len(omega)):
        x_real[i] = np.vectorize(x_f)(t, omega[i]).real
        x_imag[i] = np.vectorize(x_f)(t, omega[i]).imag
        y_real[i] = np.vectorize(y_f)(t, omega[i]).real
        y_imag[i] = np.vectorize(y_f)(t, omega[i]).imag

    plt.figure(figsize=(15, 7))

    plt.subplot(2, 2, 1)
    for i in range(len(omega)):
        plt.plot(t, x_real[i], color=color[i])
    plt.ylabel('$x_R(t)$')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    for i in range(len(omega)):
        plt.plot(t, x_imag[i], color=color[i])
    plt.ylabel('$x_I(t)$')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    for i in range(len(omega)):
        plt.plot(t, y_real[i], color=color[i])
    plt.ylabel('$y_R(t)$')
    plt.xlabel('$t$')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    for i in range(len(omega)):
        plt.plot(t, y_imag[i], color=color[i])
    plt.ylabel('$y_I(t)$')
    plt.xlabel('$t$')
    plt.grid(True)

    plt.show()

    omega_range = np.linspace(-4 * pi, 4 * pi, 1000)
    transfer_f_vals = abs(transfer_function(1j * omega_range))

    plt.figure(figsize=(8, 4))
    plt.plot(omega_range, transfer_f_vals, '-k')
    for i in range(len(omega)):
        plt.plot(abs(omega[i]), abs(transfer_function(omega[i])), color=color[i], marker='o')
    plt.ylabel('$|H(s)|$')
    plt.xlabel('$\\omega$')
    plt.show()


if __name__ == '__main__':
    a_8_d()
