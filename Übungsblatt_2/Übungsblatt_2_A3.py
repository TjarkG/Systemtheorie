import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from Übungsblatt_2_A5 import a_5_printer


def u(time):
    if time < 0:
        return 0
    else:
        return 1


def delta(t, dt):
    if t == 0:
        return 1 / dt
    return 0


def h_1(t, _):
    return 0.4 ** t * u(t)


def h_2(t, dt):
    return delta(t, dt) - delta(t - 1, dt)


def x_e(t, _):
    return u(t + 3)


def a_3_printer(t, y, label = '$y(t)$'):
    plt.figure(figsize=(10, 5))

    x_limits = [-10.5, 10.5]

    plt.plot(t, y, 'g-')
    plt.ylabel(label)
    plt.xlim(x_limits)
    if max(y) > 20000:
        plt.ylim(0, 200)
    plt.grid(True)

    plt.show()


def a_3():
    # a_5_printer(h_1)
    # a_5_printer(h_2)
    # a_5_printer(x_e)

    t = np.linspace(-100, 100, int(2e5 + 1))
    # The timescale MUST be symmetric about t=0, otherwise the function scipy.signal.convolve() will give wrong results!
    dt = (t[-1] - t[0]) / (t.size - 1)  # time step: small enough to simulate integration

    x_a = np.zeros_like(t)
    h_1_a = np.zeros_like(t)
    h_2_a = np.zeros_like(t)
    for i in range(t.size):
        x_a[i] = x_e(t[i], dt)
        h_1_a[i] = h_1(t[i], dt)
        h_2_a[i] = h_2(t[i], dt)

    y_1 = signal.convolve(x_a, signal.convolve(h_1_a, h_2_a, 'same') * dt, 'same') * dt
    y_2 = signal.convolve(signal.convolve(x_a, h_1_a, 'same') * dt, h_2_a, 'same') * dt
    y_3 = signal.convolve(x_a, signal.convolve(h_2_a, h_1_a, 'same') * dt, 'same') * dt
    y_4 = signal.convolve(signal.convolve(x_a, h_2_a, 'same') * dt, h_1_a, 'same') * dt
    h_ges = signal.convolve(h_1_a, h_2_a, 'same') * dt

    err_1 = max(y_2 - y_1)
    err_2 = max(y_3 - y_1)
    err_3 = max(y_4 - y_1)

    print(err_1, "\t", err_2, "\t", err_3)

    # Plotte Ausgangssignal
    a_3_printer(t, y_1)

    # Plotte gesamte Impulsantwort
    # kausal, BIBO stabil, hat Gedächtnis
    a_3_printer(t, h_ges, '$h_{ges}(t)$')


if __name__ == '__main__':
    a_3()
