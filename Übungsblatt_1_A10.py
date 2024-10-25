import numpy as np
import matplotlib.pyplot as plt


def x_ct(t) -> float:
    if t < -2:
        return 0.0
    if t < -1:
        return 2.0 + t
    if t < 0:
        return 1.0
    if t < 1:
        return -1.0
    if t < 2:
        return -2.0 + t
    else:
        return 0.0


def x_dt(n: int):
    if n == -3 or n == 3:
        return -2
    if n == -2 or n == 2:
        return -1
    if n == -1 or n == 1:
        return 1
    if n == 0:
        return 2
    return 0


def plot_ct(func, label):
    t = np.arange(-5, 5, 0.01)

    x = np.zeros(len(t))

    for i in range(t.size):
        x[i] = func(t[i])

    # graphical presentation of the signals:
    y_min = min(x) - 1
    y_max = max(x) + 1
    plt.figure(figsize=(10, 5))

    plt.plot(t, x, '-r', label=label)
    plt.legend()

    plt.xlabel('$t$')
    plt.ylabel(label)
    plt.axis((float(t[0]), float(t[-1]), y_min, y_max))
    plt.grid(True)

    plt.show()


def plot_dt(func, label):
    span = 6
    x = np.zeros(span * 2 + 1)
    x_axis = np.zeros(span * 2 + 1)
    for i in range(span * 2 + 1):
        x_axis[i] = i - span
        x[i] = func(i - span)

    plt.figure(figsize=(10, 8))
    area = (float(x_axis[0]) - 0.2, float(x_axis[-1]) + 0.2, min(x) - 1, max(x) + 1)

    plt.stem(x_axis, x)
    plt.xlabel('$n$')
    plt.ylabel(label)
    plt.axis(area)
    plt.grid(True)

    plt.show()


def u(t):
    if t < 0:
        return 0
    else:
        return 1


def delta_dt(n):
    if n == 0:
        return 1
    else:
        return 0


def a_10():
    plot_ct(x_ct, '$x(t)$')

    plot_ct(lambda t: x_ct(t - 1), '$x_a(t)$')
    plot_ct(lambda t: x_ct(2 - t), '$x_b(t)$')
    plot_ct(lambda t: x_ct(2 * t + 1), '$x_c(t)$')
    plot_ct(lambda t: x_ct(1 - 0.5 * t), '$x_d(t)$')
    plot_ct(lambda t: (x_ct(t) - x_ct(-t)) * u(t), '$x_e(t)$')

    plot_dt(x_dt, '$x[n]$')

    plot_dt(lambda n: x_dt(-3 - n), '$x_f[n]$')
    plot_dt(lambda n: x_dt(3 * n), '$x_g[n]$')
    plot_dt(lambda n: x_dt(n) * u(3 - n), '$x_h[n]$')
    plot_dt(lambda n: x_dt(n - 2) * delta_dt(n - 2), '$x_i[n]$')
    plot_dt(lambda n: 0.5 * x_dt(n) + 0.5 * (-1) ** n * x_dt(n), '$x_j[n]$')


if __name__ == '__main__':
    a_10()
