import matplotlib.pyplot as plt
import numpy as np
y = np.array([[1],[0]])
#fra piazza
xiN3 = 6.89684861937
xiN3_2 = 3.653753736219229

alpha = np.array([0.86, 0.59, 0.0167])

def f(xi, y, n):
    """
    Vector form of the Lane Embden equations. Modified with the correct value of xi' for xi = 0 (from piazza)
    :param xi: Previous value of xi
    :param y: Array with the previous values
    :param n: Polytropic index
    :return: Values used for the next values of y
    """
    if xi == 0:
        return np.array([y[1], -1/3])
    else:
        return np.array([y[1], -np.abs(y[0])**n - 2*y[1]/xi])


def euler(y, f, n, h):
    """
    Numerically solves differential equations with the Euler-method
    :param y: Vector with initial values for theta and chi respectively
    :param f: Stepfunction used for the numerical approximation
    :param n: Polytropic index
    :param h: Steplenght
    :return: The numerical solution to the system of differential equations with the euler-method
    """
    Y = y
    xi = np.array([0])
    switch = True
    while switch:
        F = f(xi[-1], Y[:, -1], n)      #Y, last item of every array in Y
        placeholder = Y[:,-1] + h*F
        Y = np.hstack((Y, [[item] for item in placeholder]))
        xi = np.append(xi, xi[-1]+h)
        if Y[0, -1]*Y[0, -2]<0:
            switch = False
    Y = [item[:-1] for item in Y]   #removes the last item in every array in Y
    return xi[:-1], Y


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
    S = 1/6*(k1 + 2*k2 + 2*k3 + k4)
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
    Y = y
    xi = np.array([0])
    switch = True
    while switch:
        F = RK4_step(xi[-1], Y[:, -1], f, h, n)
        Y = np.hstack((Y, [[item] for item in F]))
        xi = np.append(xi, xi[-1]+h)
        if Y[0, -1]*Y[0, -2]<0:
            switch = False
    Y = [item[:-1] for item in Y]  # removes the last item in every array in Y
    return xi[:-1],Y


def euler_error(y, f, n, h, N):
    """
    Numerically solves differential equations with the Euler-method
    This function is ment for the error calculation, such that the last value of xi is either xiN3_2 or xiN3
    :param y: Vector with initial values for theta and chi respectively
    :param f: Stepfunction used for the numerical approximation
    :param n: Polytropic index
    :param h: Steplenght
    :return: The numerical solution to the system of differential equations with the euler-method
    """
    Y = y
    xi = np.array([0])
    for i in range(N):
        F = f(xi[-1], Y[:, -1], n)      #Y, last item of every array in Y
        placeholder = Y[:,-1] + h*F
        Y = np.hstack((Y, [[item] for item in placeholder]))
        xi = np.append(xi, xi[-1]+h)
    return xi, Y


def RK4_method_error(y, f, n, h, N):
    """
    Calls on RK4_step() to calculate the next values of theta and xi for each iteration
    This function is ment for the error calculation, such that the last value of xi is either xiN3_2 or xiN3
    :param y: Vector with initial values for theta and chi respectively
    :param f: Stepfunction used for the numerical approximation
    :param n: Polytropic index
    :param h: Steplenght
    :return: The numerical solution to the system of differential equations with the RK4-method
    """
    Y = y
    xi = np.array([0])
    for i in range(N):
        F = RK4_step(xi[-1], Y[:, -1], f, h, n)
        Y = np.hstack((Y, [[item] for item in F]))
        xi = np.append(xi, xi[-1]+h)
    return xi, Y


def error_funk(n0, n1, N, xi_N, n):
    """
    Function which evaluates the error
    :param n0: Starting-step
    :param n1: End-step
    :param N: Stepsize
    :param xi_N: The last xi-value we want to hit
    :return: Arrays with the theta-values from euler and RK4, and the very last xi-value used in the evaluation in euler and RK4
    """
    Euler_err = np.array([1])
    RK4_err = np.array([1])
    xi1, xi2 = np.array(1), np.array(1)
    N_i = np.arange(n0, n1, N)
    N_i = N_i[::-1]
    h_i = xi_N/N_i
    for i in range(len(h_i)):
        xi1, Y1 = euler_error(y, f, n, h_i[i], N_i[i])
        xi2, Y2 = RK4_method_error(y, f, n, h_i[i], N_i[i])
        print(xi1[-1], xi2[-1])
        Euler_err = np.append(Euler_err, np.abs(Y1[0][-1]))
        RK4_err = np.append(RK4_err, np.abs(Y2[0][-1]))
    return Euler_err[1:], RK4_err[1:], h_i, xi1[-1], xi2[-1]


def pfunc(x, p, alpha): return np.array(-0.5*alpha*x*(1+p)*(1+3*p)*(1-x**2*alpha)**-1)
def analy_p(x, alpha): return ((np.sqrt(1-alpha)-np.sqrt(1-alpha*x**2))/(np.sqrt(1-alpha*x**2)-3*np.sqrt(1-alpha)))/ \
                              ((np.sqrt(1-alpha)-1)/(1-3*np.sqrt(1-alpha)))


def P(alpha, h, switch=False):
    """
    Solves the TOV-equations numerically, and compares with the analytical result
    :param alpha: Array-like with the values of alpha we want to plot for
    :param h: Stepsize
    :return: Nothing, plots the graphs
    """
    for i in range(len(alpha)):
        p0 = np.array([[(np.sqrt(1 - alpha[i]) - 1) / (1 - 3 * np.sqrt(1 - alpha[i]))]])
        print(p0)
        xi1, Y1 = euler(p0, pfunc, alpha[i], h)
        xi2, Y2 = RK4_method(p0, pfunc, alpha[i], h)
        plt.figure()
        plt.plot(xi1, Y1[0]/p0[0][0], '-.', label="Euler")
        plt.plot(xi2, Y2[0]/p0[0][0],'-.', label="RK4")
        plt.plot(xi1, analy_p(xi1, alpha[i]),'-.', label="Analyisk")
        if switch:
            plt.plot(xi1, 0.25*alpha[i]*(1-xi1**2)/p0[0][0], '-.', label="Newtonian approximation")
        plt.legend(loc='best')
        plt.xlabel("x-axis")
        plt.ylabel("P-axis")
        tittel = "h = " + str(h) + ", alpha = " + str(alpha[i])
        plt.title(tittel)
        plt.show()


def plot3_d_f(method, title, n=1, y=y, f=f):
    """
    Plot-function for task 3d and 3f
    :param method: The numerical method used
    :param title: The title for the plot
    :param n: The polytropic index, is 1 for these tasks
    :param y: The initial conditions
    :param f: The differential equations for the system
    :return: Nothing, plots the graphs
    """
    plt.figure()
    xi, Y= method(y, f, n, 0.5)
    plt.plot(xi, Y[0],'-.', label="h=0.5")
    xi, Y= method(y, f, n, 0.1)
    plt.plot(xi, Y[0], '-.',label="h=0.1")
    xi, Y= method(y, f, n, 0.05)
    plt.plot(xi, Y[0], '-.',label="h=0.05")
    xi, Y= method(y, f, n, 0.001)
    plt.plot(xi, Y[0], '-.',label="h=0.001")
    plt.plot(xi, np.sin(xi) / xi, '-.',label="Analytic")
    plt.title(title)
    plt.legend(loc='best')
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\theta(\xi)$")
    plt.show()
def plt3_e(h):
    """
    Solves Lane-Emden equations numerically
    :param h: Stepsize
    :return: Nothing, plots the graphs
    """
    xi1, Y1 = euler(y, f, 3/2, h)
    xi2, Y2 = euler(y, f, 3, h)
    plt.figure()
    plt.title("Nummerisk løsning for n=3/2 og n =3")
    plt.plot(xi1, Y1[0], '-.', label="Ikke-relativistiske tilfellet (n=3/2)")
    plt.plot(xi2, Y2[0], '-.', label="Ultrarelativistiske tilfellet (n=3)")
    plt.legend(loc='best')
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\theta(\xi)$")
    plt.show()
def plot3_1g(n0, n1, N):
    """
    Plot-function for the ultra-relativistic case in task 3g
    :param n0: Starting-step
    :param n1: End-step
    :param N: Stepsice
    :return: Nothing, plots the graph
    """
    Euler_err, RK4_err, h_i, xi1, xi2 = error_funk(n0, n1, N, xiN3, 3)
    plt.figure()
    plt.plot(h_i, Euler_err, '-.', label="Ultrarelativistisk euler error")
    plt.plot(h_i, RK4_err, '-.', label="Ultrarelativistisk RK4 error")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("Theta error")
    plt.xlabel("Trinnlengde h")
    plt.title("Error-plot for RK4 og Euler")
    plt.show()
def plot3_2g(n0, n1, N):
    """
    Plot-function for the non-relativistic case in task 3g
    :param n0: Starting-step
    :param n1: End-step
    :param N: Stepsize
    :return: Nothing, plots the graph
    """
    Euler_err, RK4_err, h_i, xi1, xi2 = error_funk(n0, n1, N, xiN3_2, 3/2)
    plt.figure()
    plt.plot(h_i, Euler_err, '-.', label="Ikke-relativistisk euler error")
    plt.plot(h_i, RK4_err, '-.', label="Ikke-relativistisk RK4 error")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("Theta error")
    plt.xlabel("Trinnlengde h")
    plt.title("Error-plot for RK4 og Euler")
    plt.show()


#Plotter oppgave 3d
#plot3_d_f(euler, "Numerical solution with the Euler-method, n=1")

#Plotter oppgave 3e
#plt3_e(0.0001)

#Plotter oppgave 3f
#plot3_d_f(RK4_method, "Numerical solution with the RK4-method, n=1")

#Plotter oppgave 3g
#plot3_1g(10, 1000, 1)
#plot3_2g(10, 1000, 1)

#Plotter oppgave 3i
#P(alpha, 0.001)

#Plotter oppgave 3j
#P(alpha, 0.001, True)