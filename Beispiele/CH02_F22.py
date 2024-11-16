import numpy as np
from numpy import exp
import matplotlib.pyplot as plt

a = 1.0


def u(time):
    if time < 0:
        return 0
    return 1


def transfer_function(s):
    return a / (a + s)


def x_f(time, s):
    return exp(s * time) * u(time)


def y_f(time, s):
    return transfer_function(s) * (1.0 - exp(-(s + a) * time)) * x_f(time, s)


sigma_range = np.linspace(-4 * a, 4 * a, 500)
H = transfer_function(sigma_range)

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

plt.figure(figsize=(13, 9))

plt.subplot(2, 2, 2)
for i in range(len(sigma)):
    plt.plot(t, x[i], color=color[i], linestyle='-')
plt.ylim([-0.2, 3])
plt.ylabel('$x(t)$')
plt.xlabel('$t$')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(sigma_range, H, 'k-')
for i in range(len(sigma)):
    plt.plot(sigma[i], transfer_function(sigma[i]), color=color[i], marker='o')
plt.ylim([transfer_function(sigma[1]) - 1, transfer_function(sigma[2]) + 1])
plt.ylabel('$H(s)$')
plt.xlabel('$s = \\sigma$')
plt.grid(True)

plt.subplot(2, 2, 4)
for i in range(len(sigma)):
    plt.plot(t, y[i], color=color[i], linestyle='-')
plt.ylim([-0.2, 3])
plt.ylabel('$y(t)$')
plt.xlabel('$t$')
plt.grid(True)

plt.show()
