import matplotlib.pyplot as plt
import numpy as np


def u(time: float) -> float:
    if time < 0:
        return 0
    else:
        return 1


R = 1e6
C = 1e-6
a = 1 / (R * C)


def y(t):
    s = -a * (1 + 1e-2)
    return a / (a + s) * (np.exp(s * t) - np.exp(-a * t)) * u(t)


def y_a(t):
    return a * t * np.exp(- a * t) * u(t)


def a_7_b():
    t = np.arange(-1, 8, 0.01)

    y_1 = np.vectorize(y_a)(t)
    y_2 = np.vectorize(y)(t)

    plt.figure(figsize=(10, 5))

    plt.plot(t, y_1)
    plt.plot(t, y_2)

    plt.ylim(-0.05, 0.5)
    plt.xlabel('$t$')
    plt.ylabel('$y(t)$')
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    a_7_b()
