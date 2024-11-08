import numpy as np
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


def a_1_printer(h_f, x_limits=None):
    if x_limits is None:
        x_limits = [-10.5, 50.5]
    n = np.linspace(-100, 100, 201, dtype='int')
    # The timescale MUST be symmetric about n=0, otherwise the function scipy.signal.convolve() will give wrong results!

    h = np.zeros(n.size)
    for i in range(n.size):
        h[i] = h_f(n[i])

    plt.figure(figsize=(10, 5))

    plt.stem(n, h)
    plt.ylabel('$h[n]$')
    plt.xlim(x_limits)
    if max(h) > 200:
        plt.ylim(0, 200)
    plt.grid(True)

    plt.show()


def a_1(part):
    match part:
        case 'a':
            def h(n):
                return (1 / 5) ** n * u(n)

            # Kausal, BIBO Stabil
            a_1_printer(h)

        case 'b':
            def h(n):
                return 0.8 ** n * u(n + 2)

            # nicht kausal, BIBO Stabil
            a_1_printer(h)

        case 'c':
            def h(n):
                return 0.5 ** n * u(-n)

            # nicht kausal, nicht BIBO Stabil
            a_1_printer(h, [-20.5, 20.5])

        case 'd':
            def h(n):
                return 5.0 ** n * u(3 - n)

            # nicht kausal, BIBO Stabil
            a_1_printer(h)

        case 'e':
            def h(n):
                return n * (0.2 ** n) * u(n - 1)

            # Kausal, BIBO Stabil
            a_1_printer(h)

        case 'f':
            def h(n):
                return (-0.4) ** n * u(n) + 1.01 ** n * u(1 - n)

            # nicht kausal, nicht BIBO Stabil
            a_1_printer(h, [-20.5, 10.5])


if __name__ == '__main__':
    a_1(input("Part: "))
