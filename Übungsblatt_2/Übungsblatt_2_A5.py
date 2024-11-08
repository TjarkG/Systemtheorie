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
    if t == 1:
        return 1 / dt
    return 0


def a_1_printer(h_f):
    t = np.linspace(-100, 100, int(2e5 + 1))
    # The timescale MUST be symmetric about t=0, otherwise the function scipy.signal.convolve() will give wrong results!
    dt = (t[-1] - t[0]) / (t.size - 1)  # time step: small enough to simulate integration

    h = np.zeros_like(t)
    for i in range(t.size):
        h[i] = h_f(t[i], dt)

    plt.figure(figsize=(10, 5))

    x_limits = [-10.5, 50.5]

    plt.plot(t, h, 'g-')
    plt.ylabel('$h(t)$')
    plt.xlim(x_limits)
    if max(h) > 200:
        plt.ylim(0, 200)
    plt.grid(True)

    plt.show()


def a_2(part):
    match part:
        case 'a':
            def h(t, _):
                return math.exp(-4 * t) * u(t - 2)

            # Kausal, BIBO Stabil
            a_1_printer(h)

        case 'b':
            def h(t, _):
                return math.exp(-5 * t) * u(3 - t)

            # nicht kausal, nicht BIBO Stabil
            a_1_printer(h)

        case 'c':
            def h(t, _):
                return math.exp(6 * t) * u(-1 - t)

            # nicht kausal, BIBO Stabil
            a_1_printer(h)

        case 'd':
            def h(t, _):
                return math.exp(-6 * abs(t))

            # nicht kausal, BIBO Stabil
            a_1_printer(h)

        case 'e':
            def h(t, _):
                return t * math.exp(-t) * u(t)

            # kausal, BIBO Stabil
            a_1_printer(h)

        case 'f':
            def h(t, _):
                return (2 * math.exp(-t) - math.exp((t - 100) / 100)) * u(t)

            # kausal, nicht BIBO Stabil
            a_1_printer(h)


if __name__ == '__main__':
    a_2(input("Part: "))
