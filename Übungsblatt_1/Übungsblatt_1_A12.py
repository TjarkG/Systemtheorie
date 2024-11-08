import numpy as np
import matplotlib.pyplot as plt
from cmath import *


def plotPolar_ct(function, points):
    val = function(points)
    phi = [phase(i) for i in val]
    r = [abs(i) for i in val]

    plt.figure(dpi=200)
    plt.polar(phi, r)
    plt.show()


def plotPolar_dt(function, points, format='x'):
    val = function(points)
    phi = [phase(i) for i in val]
    r = [abs(i) for i in val]

    plt.figure(dpi=200)
    plt.polar(phi, r, format)
    plt.show()


def findPeriode(function) -> int:
    # Startwert
    x_0 = function(0)
    n = 1
    for zeros in range(1, 100):
        # Suche Wiederholung
        while not isclose(x_0, function(n)):
            n += 1
            # Keine Wiederholung gefunden
            if n > 1000:
                return -1

        # Überprüfe
        for i in range(1, n):
            if not isclose(function(i), function(i + n)):
                break
        else:
            return n
    # Keine der Wiederholungen von x_0 ist periodisch
    return -2


def a_12():
    match input('Part: '):
        case 'a':
            f = lambda t: 1j * e ** (1j * 10 * t)
            plotPolar_ct(f, np.arange(0, 2 * pi, 0.01))
        case 'b':
            f = lambda t: e ** ((-1 / 5 + 1j) * t)
            plotPolar_ct(f, np.arange(0, 6 * pi, 0.01))
        case 'c':
            f = lambda n: e ** (1j * pi * n / 7)
            print(findPeriode(f))
            plotPolar_dt(f, np.arange(14))
        case 'd':
            f = lambda n: 3 * e ** (3j * pi * (n + 1 / 2) / 5)
            print(findPeriode(f))
            plotPolar_dt(f, np.arange(10))
        case 'e':
            f = lambda n: 1 + e ** (4j * pi * n / 7) - e ** (2j * pi * n / 5)
            print(findPeriode(f))
            plotPolar_dt(f, np.arange(35), '-x')


if __name__ == '__main__':
    a_12()
