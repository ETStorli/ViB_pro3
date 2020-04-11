import matplotlib.pyplot as plt
import numpy as np

def f(theta, chi, xi, n): return np.array([chi, -theta**n - 2*chi/xi])


def plot(t, plot, title):
    """
    Very useful plotting function, both useful for plot with one graph and multiple graphs
    :param t: Array-like with time values for the plot
    :param plot: Array-like with pairs of what is to be plotted along y-axis and label. i.g. [y, "position", z, "sinus"]
    :param title: Title for the plot
    :return: nothing, shows the graph at the end
    """
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\theta$")
    plt.title(title)
    for i in range(0, len(plot), 2):
        plt.plot(t, plot[i], label=plot[i+1])
    plt.legend(loc="best")
    plt.show()

def euler(y, f, n, h):
    """
    :param y: Vector with initial values for theta and chi
    :param f: Stepfunction used for the nummerical approximation
    :param n: Polytropic index
    :param h: Steplenght
    """
    theta = np.array([y[0]])
    chi = np.array([y[1]])
    xi = np.array([.001])
    switch = True
    while switch:
        F = f(theta[-1], chi[-1], xi[-1], n)
        theta = np.append(theta, theta[-1] + h*F[0])
        chi = np.append(chi, chi[-1] + h*F[1])
        xi = np.append(xi, xi[-1]+h)
        #print("1")
        if theta[-1]*theta[-2]<0:
            switch = False
    return xi, theta, chi

y = np.array([1,0])

xi1, theta1, _ = euler(y, f, 1, .5)
xi2, theta2, _ = euler(y, f, 1, .1)
xi3, theta3, _ = euler(y, f, 1, .01)
xi4, theta4, _ = euler(y, f, 1, .001)

plt.plot(xi1, theta1, label ="h=.5")
plt.plot(xi2, theta2, label="h=.1")
plt.plot(xi3, theta3, label="h=.01")
plt.plot(xi4, theta4, label="h=.001")
plt.plot(xi4, np.sin(xi4)/xi4, label="analytisk")
plt.legend(loc="best")
plt.show()


def RK4_step():
    "one step of Runge-Kutta"

def RK4():
    "Calls on RK4_step() to find the values for each iteration to solve the system of differential equations"
