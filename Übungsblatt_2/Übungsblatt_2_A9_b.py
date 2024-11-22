import numpy as np
import matplotlib.pyplot as plt

R = 1e6
C = 1e-6
a = 1 / (R * C)

t_min = -1.0
t_max = 10.0
n = 1002
tau = (t_max - t_min) / n


def u(time):
    if time < 0:
        return 0
    else:
        return 1

def delta(t):
    if t == 0:
        return 1
    return 0

def impulse_response(t):
    if 0 > t:
        return 0
    return 1 / ((1 + a * tau) ** (t + 1)) * u(t) - 1 / ((1 + a * tau) ** t) * u(t - 1)
    # return 1 / (1 + a * tau) * delta(t) - (a * tau / ((1 + a * tau) ** (n + 1))) * u(t - 1)
    # return ((- a * tau) ** u(n - 1)) / ((1 + a * tau) ** (n + 1)) * u(t)


def a_8_d():
    t = np.linspace(t_min, t_max, n)
    t_2 = np.linspace(t_min / tau, t_max / tau, n)
    transfer_f_vals = np.zeros_like(t_2)

    for i in range(t_2.size):
        transfer_f_vals[i] = impulse_response(t_2[i])

    plt.figure(figsize=(8, 4))
    plt.plot(t, transfer_f_vals, '-k')
    plt.ylabel('$h(n)$')
    plt.xlabel('$n$')
    plt.ylim(-0.02, 0.02)
    plt.show()


if __name__ == '__main__':
    a_8_d()
