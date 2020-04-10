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
    plt.xlabel("Tid [$s$]")
    plt.ylabel("Utslag")
    plt.title(title)
    for i in range(0, len(plot), 2):
        plt.plot(t, plot[i], label=plot[i+1])
    plt.legend(loc="best")
    plt.show()

def euler(y, f, n, h=h):
    """
    :param y: Vector with initial values for theta and chi
    :param f: Stepfunction used for the nummerical approximation
    :param h: Steplenght
    """
    n = int(N/h)
    theta, chi = np.zeros(n), np.zeros(n)
    theta[0] = y[0]
    chi[0] = y[1]
    xi = np.linspace(0,N,n)
    for i in range(n-1):
        F = f(theta[i], chi[i], xi[i], n)
        theta[i+1] = theta[i] + h*F[0]
        chi[i + 1] = chi[i] + h*F[1]
    return xi, theta, chi

def RK4_step():
    "one step of Runge-Kutta"

def RK4():
    "Calls on RK4_step() to find the values for each iteration to solve the system of differential equations"
