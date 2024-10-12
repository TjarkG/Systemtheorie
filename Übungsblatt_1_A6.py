from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


# plot list of arrays
def plot(x_axis: np.array, arrays: list[Tuple[np.array, str]], y_min, y_max):
    plt.figure(figsize=(10, 8))
    area = (x_axis[0] - 0.2, x_axis[-1] + 0.2, y_min, y_max)

    for i in range(len(arrays)):
        plt.subplot(3, 1, i + 1)
        plt.stem(x_axis, arrays[i][0])
        plt.ylabel(arrays[i][1])
        plt.axis(area)
        plt.grid(True)

    plt.xlabel('$n$')
    plt.show()


def a_6_a_b(n, x):
    assert len(n) == len(x)
    x_even = np.empty_like(x)
    x_odd = np.empty_like(x)

    for i in range(len(x)):
        x_even[i] = (x[i] + x[len(x) - 1 - i]) / 2
        x_odd[i] = (x[i] - x[len(x) - 1 - i]) / 2

    e = 0.0
    e_even = 0.0
    e_odd = 0.0
    for i in range(len(x)):
        e += x[i] ** 2
        e_even += x_even[i] ** 2
        e_odd += x_odd[i] ** 2

    print("E = ", e)
    print("E_even = ", e_even)
    print("E_odd = ", e_odd)

    plot(n, [(x, '$x[n]$'), (x_even, '$x_{even}[n]$'), (x_odd, '$x_{odd}[n]$')], -2, 4)


def a_6_c_d(t, x_f):
    def x_even_f(time):
        return (x_f(time) + x_f(-time)) / 2

    def x_odd_f(time):
        return (x_f(time) - x_f(-time)) / 2

    x = np.empty_like(t)
    x_even = np.empty_like(t)
    x_odd = np.empty_like(t)
    for i in range(t.size):
        x[i] = x_f(t[i])
        x_even[i] = x_even_f(t[i])
        x_odd[i] = x_odd_f(t[i])

    # graphical presentation of the signals:
    y_min = min(min(x), min(x_even), min(x_odd)) - 1
    y_max = max(max(x), max(x_even), max(x_odd)) + 1
    plt.figure(figsize=(10, 5))

    plt.plot(t, x, '-r', label='$x(t)$')
    plt.plot(t, x_even, '--g', label='$x_{even}(t)$')
    plt.plot(t, x_odd, '-.b', label='$x_{odd}(t)$')
    plt.legend()

    plt.xlabel('$t$')
    plt.ylabel('$x(t)$')
    plt.axis((float(t[0]), float(t[-1]), y_min, y_max))
    plt.grid(True)

    plt.show()


def a_6(part):
    match part:
        case 'a':
            n = np.arange(-5, 5 + 1)
            x = np.array([0, 0, 0, 1, 2, 3, 0, 0, 2, 0, 0])
            a_6_a_b(n, x)
        case 'b':
            n = np.arange(-6, 6 + 1)
            x = np.array([0, 0, -1, 2, 2, 1, 1, 2, 1, -1, 0, 0, 0])
            a_6_a_b(n, x)
        case 'c':
            t = np.arange(-3, 3, 0.01)

            def x_f(time):
                if time < 0:
                    return 0
                if time < 1:
                    return time
                if time < 2:
                    return 2 - time
                else:
                    return 0
            a_6_c_d(t, x_f)
        case 'd':
            t = np.arange(-3, 3, 0.01)

            def x_f(time):
                if time < -1:
                    return 2
                if time < 0:
                    return -2*time
                if time < 1:
                    return time
                else:
                    return 1
            a_6_c_d(t, x_f)


if __name__ == '__main__':
    a_6('d')
