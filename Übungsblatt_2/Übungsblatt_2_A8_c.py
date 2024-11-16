import numpy as np
import matplotlib.pyplot as plt

R = 10
C = 10


def u(t):
    if t < 0:
        return 0
    return 1


def transfer_function(s):
    return 1 / (R + C * s)


def x_f(t, s):
    return np.exp(s * t) * u(t)


def y_f(t, s):
    return 1 / (R + C * s) * (np.exp(s * t) - np.exp(-R / C * t)) * u(t)


def a_8_c():
    a = R / C

    t = np.linspace(-1.0, 10.0, 1000)

    dev = 0.1
    sigma = [-3.0 * a,  # negative value
             -(1.0 + dev) * a, -(1.0 - dev) * a,  # points near pole
             -dev * a, 0.0, dev * a,  # points near zero
             1.0 * a]  # positive value

    color = ['gray', 'black', 'blue', 'indigo', 'green', 'pink', 'red']

    x = np.zeros((len(sigma), t.size))
    y = np.zeros((len(sigma), t.size))

    for i in range(len(sigma)):
        x[i] = np.vectorize(x_f)(t, sigma[i])
        y[i] = np.vectorize(y_f)(t, sigma[i])

    plt.figure(figsize=(8, 12))

    plt.subplot(3, 1, 1)
    for i in range(len(sigma)):
        plt.plot(t, x[i], color=color[i], linestyle='-')
    plt.ylim([-0.2, 3])
    plt.ylabel('$x(t)$')
    plt.xlabel('$t$')
    plt.grid(True)

    # Plot Transfer Function
    sigma_range = np.linspace(-4 * a, 4 * a, 500)
    transfer_f_vals = transfer_function(sigma_range)

    plt.subplot(3, 1, 2)
    plt.plot(sigma_range, transfer_f_vals, 'k-')
    # Plot Points on Transfer Function
    for i in range(len(sigma)):
        plt.plot(sigma[i], transfer_function(sigma[i]), color=color[i], marker='o')
    plt.ylim([-1.2, 1.2])
    plt.ylabel('$H(s)$')
    plt.xlabel('$s = \\sigma$')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    for i in range(len(sigma)):
        plt.plot(t, y[i], color=color[i], linestyle='-')
    plt.ylim([-0.05, 0.2])
    plt.ylabel('$y(t)$')
    plt.xlabel('$t$')
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    a_8_c()
