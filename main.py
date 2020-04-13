import matplotlib.pyplot as plt
import numpy as np
y = np.array([[1],[0]])
#fra piazza
xiN3 = 6.89684861937
xiN3_2 = 3.653753736219229


def f(xi, y, n): return np.array([y[1], -np.abs(y[0])**n - 2*y[1]/xi])


def euler(y, f, n, h):
    """
    :param y: Vector with initial values for theta and chi respectively
    :param f: Stepfunction used for the numerical approximation
    :param n: Polytropic index
    :param h: Steplenght
    :return: The numerical solution to the system of differential equations with the euler-method
    """
    Y = y
    xi = np.array([.001])
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
    Y = y
    xi = np.array([.001])
    switch = True
    while switch:
        F = RK4_step(xi[-1], Y[:, -1], f, h, n)
        Y = np.hstack((Y, [[item] for item in F]))
        xi = np.append(xi, xi[-1]+h)
        if Y[0, -1]*Y[0, -2]<0:
            switch = False
    Y = [item[:-1] for item in Y]  # removes the last item in every array in Y
    return xi[:-1],Y


def error_funk(n0, n1, N, xi_N):
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
    h_i = xi_N/N_i[::-1]
    for h in h_i:
        print(h)
        theta1, xi1, _ = euler(y, f, 3, h)
        theta2, xi2, _ = RK4_method(y, f, 3, h)
        Euler_err = np.append(Euler_err, np.abs(theta1[-1]))
        RK4_err = np.append(RK4_err, np.abs(theta2[-1]))
    return Euler_err[1:], RK4_err[1:], h_i, xi1[-1], xi2[-1]

def pfunc(x, p, alpha): return np.array(-0.5*alpha*x*(1+p)*(1+3*p)*(1-x**2*alpha)**-1)

def solv_p(alph, pfunc, h):
    """

    :param alph: liste med 3 alpha verdier, verdt å huske at vi da må iterere gjennom den og
    :param pfunc: funksjon for deriverte av p, er likn 36
    :param h: trinnlengde, gikk for 0.01 for simpelhetskyld
    :return: et 3x2xN rom, hvor N er lengden på skritt som blir tatt av euler eller rk4. Mulig problem oppstår om de har ulik antall total skritt
    """
    solved_p = np.array([[1,1],[2,2],[3,3]])
    for i in range(3):
        p0 = np.array([[(np.sqrt (1 - alph[i]) - 1) / (1 - 3 * np.sqrt (1 - alph[i]))]])
        solved_p[i][0] = euler(p0, pfunc, alph[i], h)
        solved_p[i][1] = RK4_method(p0, pfunc, alph[i], h)
    return solved_p

def analy_p( x, alph): return (((np.sqrt(1-alph)-np.sqrt(1-alph*x**2))/(np.sqrt(1-alph*x**2)-3*np.sqrt(1-alph))))/ \
                              ((np.sqrt(1-alph)-1)/(1-3*np.sqrt(1-alph)))


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


def plot1_3g(n0, n1, N):
    """
    Plot-function for the ultra-relativistic case in task 3g
    :param n0: Starting-step
    :param n1: End-step
    :param N: Stepsice
    :return: Nothing, plots the graph
    """
    Euler_err, RK4_err, h_i, xi1, xi2 = error_funk(n0, n1, N, xiN3)
    plt.figure()
    plt.plot(h_i, Euler_err, '-.', label="Ultrarelativistisk euler error")
    plt.plot(h_i, RK4_err, '-.', label="Ultrarelativistisk RK4 error")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("Theta error")
    plt.xlabel("Trinnlengde h")
    plt.show()


def plot2_3g(n0, n1, N):
    """
    Plot-function for the non-relativistic case in task 3g
    :param n0: Starting-step
    :param n1: End-step
    :param N: Stepsize
    :return: Nothing, plots the graph
    """
    Euler_err, RK4_err, h_i, xi1, xi2 = error_funk(n0, n1, N, xiN3_2)
    plt.figure()
    plt.plot(h_i, Euler_err, '-.', label="Ikke-relativistisk euler error")
    plt.plot(h_i, RK4_err, '-.', label="Ikke-relativistisk RK4 error")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("Theta error")
    plt.xlabel("Trinnlengde h")
    plt.show()

def plot3_i(alph, solved_p, analy_p, h, a ):
    """

    :param alph: liste med alpha verdier
    :param solved_p: rom som inneholder løsninger for p
    :param analy_p: normalisert analytisk løsning for newton redusert approximasjon
    :param h: skritt
    :param a: boolk statement som bestemmer plot av analytisk
    :return: nada
    """
    for i in range(len(alph)):
        x = np.linspace(0, 1, 1000)
        plt.figure()
        plt.title("sammenlikning euler, rk4 og analytisk for h=", h, "og alpha = ", alph[i])
        plt.plot(x, analy_p(x, alph[i]), 'b.', label="analytisk")
        plt.plot(x, solved_p[i][0], 'r.', label="euler")
        plt.plot(x, solved_p[i][1], 'g.', label="RK 4")
        if a:
            plt.plot(x, (0.25*alph*(1-x**2)), '.', label="newtonsk approximasjon")
        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

#Plotter oppgave 3d
plot3_d_f(euler, "Numerical solution with the Euler-method")

#Plotter oppgave 3f
#plot3_d_f(RK4_method, "Numerical solution with the RK4-method")

#plot1_3g(1, 1000, 1)

# oppg 3i

alp =[0.86, 0.59, 0.0167]
solved_p = solv_p(alp, pfunc, 0.01)
plot3_i(alp, solved_p, analy_p, 0.01, False)