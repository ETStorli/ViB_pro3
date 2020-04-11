import matplotlib.pyplot as plt
import numpy as np

def f(xi, y, n): return np.array([y[1], -y[0]**n - 2*y[1]/xi])


def euler(y, f, n, h):
    """
    :param y: Vector with initial values for theta and chi respectively
    :param f: Stepfunction used for the numerical approximation
    :param n: Polytropic index
    :param h: Steplenght
    :return: The numerical solution to the system of differential equations with the euler-method
    """
    theta = np.array([y[0]])
    chi = np.array([y[1]])
    xi = np.array([.001])
    switch = True
    while switch:
        F = f(xi[-1], [theta[-1], chi[-1]], n)
        theta = np.append(theta, theta[-1] + h*F[0])
        chi = np.append(chi, chi[-1] + h*F[1])
        xi = np.append(xi, xi[-1]+h)
        if theta[-1]*theta[-2]<0:
            switch = False
    return xi, theta, chi


def RK4_step(xi, y, f, h, n):
    """
    Calculates one step of the RK4-algorithm
    :param xi: The value of xi we evaluate y at
    :param y: Array-like with the previous values of theta and chi respectively
    :param f: The function that describes the system of differential equations (y' = f, vector form)
    :param h: Step-size
    :param n: Polytropic index
    :return: Next value of theta and xi in an array
    """
    k1 = f(xi, y, n)
    k2 = f(xi + h/2, y + h*k1/2, n)
    k3 = f(xi + h/2, y + h*k2/2, n)
    k4 = f(xi + h, y + h*k3, n)
    S = 1/6*(k1 + 2*k2 + 2*k3 +k4)
    return y + h*S


def RK4_method(y, f, n, h):
    """
    Calls on RK4_step() to calculate the next values of theta and xi for each iteration
    :param y: Vector with initial values for theta and chi respectively
    :param f: Stepfunction used for the numerical approximation
    :param n: Polytropic index
    :param h: Steplenght
    :return: The numerical solution to the system of differential equations with the RK4-method
    """
    theta = np.array([y[0]])
    chi = np.array([y[1]])
    xi = np.array([.001])
    switch = True
    while switch:
        F = RK4_step(xi[-1], [theta[-1], chi[-1]], f, h, n)
        theta = np.append(theta, F[0])
        chi = np.append(chi, F[1])
        xi = np.append(xi, xi[-1]+h)
        if theta[-1]*theta[-2]<0:
            switch = False
    return xi, theta, chi


y = np.array([1,0])
n = 1

xi, theta,_  = euler(y, f, n, 0.5)
plt.plot(xi, theta, label="h=0.5")
xi, theta,_  = euler(y, f, n, 0.1)
plt.plot(xi, theta, label="h=0.1")
xi, theta,_  = euler(y, f, n, 0.01)
plt.plot(xi, theta, label="h=0.01")
xi, theta,_  = euler(y, f, n, 0.0001)
plt.plot(xi, theta, label="h=0.0001")
plt.plot(xi, np.sin(xi)/xi, label="analytisk, n=1")
plt.title("Numerical solution with ther Euler-method")
plt.legend(loc='best')
plt.xlabel(r"$\xi$")
plt.ylabel(r"$\theta$")
plt.show()
print(xi[-1], xi[-2])

xi, theta,_  = RK4_method(y, f, n, 0.5)
plt.plot(xi, theta, label="h=0.5")
xi, theta,_  = RK4_method(y, f, n, 0.1)
plt.plot(xi, theta, label="h=0.1")
xi, theta,_  = RK4_method(y, f, n, 0.01)
plt.plot(xi, theta, label="h=0.01")
xi, theta,_  = RK4_method(y, f, n, 0.0001)
plt.plot(xi, theta, label="h=0.0001")
plt.plot(xi, np.sin(xi)/xi, label="analytisk, n=1")
plt.title("Numerical solution with the RK4-method")
plt.legend(loc='best')
plt.xlabel(r"$\xi$")
plt.ylabel(r"$\theta$")
plt.show()
print(xi[-1], xi[-2])