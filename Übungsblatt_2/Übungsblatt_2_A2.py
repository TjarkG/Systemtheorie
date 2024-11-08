import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def u(time):
    if time < 0:
        return 0
    else:
        return 1


def delta(t, dt):
    if t == 0:
        return 1 / dt
    return 0


def a_1_printer(x_f, h_f, x_limits=None):
    if x_limits is None:
        x_limits = [-10.5, 10.5]

    t = np.linspace(-100, 100, int(2e5 + 1))
    # The timescale MUST be symmetric about t=0, otherwise the function scipy.signal.convolve() will give wrong results!
    dt = (t[-1] - t[0]) / (t.size - 1)  # time step: small enough to simulate integration

    x = np.zeros_like(t)
    h = np.zeros_like(t)
    for i in range(t.size):
        x[i] = x_f(t[i])
        h[i] = h_f(t[i], dt)

    y = signal.convolve(x, h, 'same') * dt  # multiplication with the time interval (integration)

    plt.figure(figsize=(8, 8))

    plt.subplot(311)
    plt.plot(t, x, 'b-')
    plt.ylabel('$x(t)$')
    plt.xlim(x_limits)
    plt.grid(True)

    plt.subplot(312)
    plt.plot(t, h, 'g-')
    plt.ylabel('$h(t)$')
    plt.xlim(x_limits)
    plt.grid(True)

    plt.subplot(313)
    plt.plot(t, y, 'r-')
    plt.ylabel('$y(t)$')
    plt.xlim(x_limits)
    plt.grid(True)
    plt.xlabel('$t$')

    plt.show()


def a_2(part):
    match part:
        case 's':
            def x(t):
                return u(t) - u(t - 10) + u(t - 20) - u(t - 30)

            def h(t, _):
                return math.exp(-t / 5) * u(t)

            a_1_printer(x, h, [-10.5, 50.5])
        case 'a':
            def x(t):
                return u(t - 3) - u(t - 5)

            def h(t, _):
                return math.exp(-3 * t) * u(t)

            a_1_printer(x, h)

        case 'b':
            def x(t):
                return u(t) - 2 * u(t - 2) + u(t - 5)

            def h(t, _):
                return math.exp(2 * t) * u(1 - t)

            a_1_printer(x, h)

        case 'c':
            def x(t):
                if 0 <= t <= 1:
                    return t + 1
                elif 1 < t <= 2:
                    return 2 - t
                return 0

            def h(t, dt):
                return -delta(t + 2, dt) + 2 * delta(t - 1, dt)

            a_1_printer(x, h, [-5.5, 5.5])


if __name__ == '__main__':
    a_2(input("Part: "))
