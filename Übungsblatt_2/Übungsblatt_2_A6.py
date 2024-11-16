import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def u(n):
    if n < 0:
        return 0
    else:
        return 1


def delta(t):
    if t == 0:
        return 1
    return 0


def h_1_f(n):
    return (-0.5) ** n * u(n)


def h_2_f(n):
    return u(n) + 0.5 * u(n - 1)


def a_6_printer(n, h):
    plt.figure(figsize=(10, 5))

    plt.stem(n, h)
    plt.ylabel('$h[n]$')
    plt.xlim(-10.5, 20.5)
    if max(h) > 50:
        plt.ylim(-1, 22)
    plt.grid(True)

    plt.show()


def a_6():
    n = np.linspace(-100, 100, 201, dtype='int')
    # The timescale MUST be symmetric about n=0, otherwise the function scipy.signal.convolve() will give wrong results!

    h_1 = np.zeros(n.size)
    h_2 = np.zeros(n.size)
    u_a = np.zeros(n.size)
    for i in range(n.size):
        h_1[i] = h_1_f(n[i])
        h_2[i] = h_2_f(n[i])
        u_a[i] = u(n[i])

    # a_6_printer(n, h_1)
    # a_6_printer(n, h_2)

    # a) Plotte Impulsantwort
    # kausal, BIBO stabil, hat Gedächtnis
    h_ges = signal.convolve(h_1, h_2, 'same')
    # a_6_printer(n, h_ges)

    # b) Sprungantwort
    # s = signal.convolve(u_a, h_ges, 'same')
    s = np.zeros(n.size)

    # Summe von minus Unendlich
    """for i in range(n.size):
        for k in range(i+1):
            s[i] += h_ges[k]"""

    # Summe von 0 - nur für kausale Systeme
    """for i in range(n.size):
        if n[i] >= 0:
            for k in range(n[i] + 1):
                s[i] += h_ges[n[k]]"""

    for i in range((n.size - 1) // 2, n.size):
        s[i] = s[i - 1] + h_ges[i]

    a_6_printer(n, s)

    # c) Impulsantwort aus Sprungantwort
    h_ges_2 = np.zeros(n.size)

    for i in range(1, n.size):
        h_ges_2[i] = s[i] - s[i - 1]

    a_6_printer(n, h_ges_2)


if __name__ == '__main__':
    a_6()
