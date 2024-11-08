import numpy as np
import matplotlib.pyplot as plt
from cmath import *


def a_7_a():
    num = complex(float(input("Enter real part: ")), float(input("Enter imaginary part: ")))

    print("Number: ", num)
    print("Conjugated: ", num.conjugate())
    print("Abs: ", abs(num))


b_numbers = np.array([1 + 1j, 1 / 2 - 1j * sqrt(3) / 2, (1 - 1j) ** 2, (1 + 1j) / (1 - 1j),
                      (sqrt(2) + sqrt(2) * 1j) / (1 + 1j * sqrt(3)), 1 / 2 * e ** (-1j * pi), e ** (1j * pi / 2),
                      sqrt(2) * e ** (-1j * pi / 4), e ** (-5j * pi / 2), e ** 1j])


def a_7_b():
    for z in b_numbers:
        print(z, abs(z), phase(z) * 180 / pi)


def a_7_c(n):
    for z in b_numbers:
        roots = [rect(pow(abs(z), 1 / n), phase(z) / n + 2 * pi * k / n) for k in range(n)]
        print(roots)


def plot_n_roots(z, n):
    r = np.zeros(n)
    phi = np.zeros(n)
    for i in range(n):
        r[i] = pow(abs(z), 1 / n)
        phi[i] = phase(z) / n + 2 * pi * i / n

    fig = plt.figure(dpi=200)
    fig.add_subplot(projection='polar')
    plt.polar(phi, r, 's')
    plt.show()


if __name__ == '__main__':
    # a_7_a()
    # a_7_b()
    # a_7_c(3)
    plot_n_roots(b_numbers[2], 4)
