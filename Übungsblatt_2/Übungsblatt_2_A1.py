import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def u(n):
    if n < 0:
        return 0
    else:
        return 1


def delta(n):
    if n == 0:
        return 1
    return 0


def a_1_printer(h_f, x_f, x_limits=None):
    if x_limits is None:
        x_limits = [-10.5, 50.5]
    n = np.linspace(-100, 100, 201, dtype='int')
    # The timescale MUST be symmetric about n=0, otherwise the function scipy.signal.convolve() will give wrong results!

    x = np.zeros(n.size)
    h = np.zeros(n.size)
    for i in range(n.size):
        x[i] = x_f(n[i])
        h[i] = h_f(n[i])

    y = signal.convolve(x, h, 'same')

    plt.figure(figsize=(8, 8))

    plt.subplot(311)
    plt.stem(n, x)
    plt.ylabel('$x[n]$')
    plt.xlim(x_limits)
    plt.grid(True)

    plt.subplot(312)
    plt.stem(n, h)
    plt.ylabel('$h[n]$')
    plt.xlim(x_limits)
    plt.grid(True)

    plt.subplot(313)
    plt.stem(n, y)
    plt.ylabel('$y[n]$')
    plt.xlim(x_limits)
    plt.grid(True)
    plt.xlabel('$n$')

    plt.show()


def a_1(part):
    match part:
        case 's':
            def x(n):
                return u(n) - u(n - 10) + u(n - 20) - u(n - 30)

            def h(n):
                return math.exp(-n / 5) * u(n)

            a_1_printer(h, x, [-10.5, 50.5])

        case 'a':
            def x(n):
                return delta(n) + 2 * delta(n - 1) - delta(n - 3)

            def h(n):
                return 2 * delta(n - 2) + 2 * delta(n + 3)

            a_1_printer(h, x,[-10.5, 10.5])

        case 'b':
            def x(n):
                return (1 / 3) ** (-n) * u(-n - 1)

            def h(n):
                return u(n + 2)

            a_1_printer(h, x,[-10.5, 10.5])

        case 'c':
            def x(n):
                return (-1 / 3) ** n * u(n - 4)

            def h(n):
                return 4.0 ** n * u(2 - n)

            a_1_printer(h, x,[-5.5, 15.5])

        case 'd':
            def x(n):
                if 3 <= n <= 8:
                    return 1
                return 0

            def h(n):
                if 4 <= n <= 15:
                    return 1
                return 0

            a_1_printer(h, x,[-0.5, 30.5])


if __name__ == '__main__':
    a_1(input("Part: "))
