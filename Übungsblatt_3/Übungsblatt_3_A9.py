import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from math import exp, cos, sin


def u(time):
    if time >= 0:
        return 1
    else:
        return 0


def delta(t, dt):
    if t == 0:
        return 1 / dt
    return 0


def a_9():
    def x_f(time):
        return exp(-abs(time))

    def h_f(time):
        return exp(-time) * cos(time) * u(time)

    def y_f(time):
        return 2 / 5 * exp(time) * u(-time) + exp(-time) * u(time) * (2 / 5 * cos(time) + 4 / 5 * sin(time))

    x_limits = [-10.5, 10.5]
    t = np.linspace(-100, 100, int(2e5 + 1))
    # The timescale MUST be symmetric about t=0, otherwise the function scipy.signal.convolve() will give wrong results!
    dt = (t[-1] - t[0]) / (t.size - 1)  # time step: small enough to simulate integration

    x = np.zeros_like(t)
    h = np.zeros_like(t)
    y_a = np.zeros_like(t)
    for i in range(t.size):
        x[i] = x_f(t[i])
        h[i] = h_f(t[i])
        y_a[i] = y_f(t[i])

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
    plt.plot(t, y_a, 'k-')
    plt.plot(t, y, 'r-')
    plt.ylabel('$y(t)$')
    plt.xlim(x_limits)
    plt.ylim(-0.05,0.5)
    plt.grid(True)
    plt.xlabel('$t$')

    plt.show()


if __name__ == '__main__':
    a_9()
