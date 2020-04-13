import matplotlib.pyplot as plt
import numpy as np

def f(xi, y, n): return np.array([y[1], -np.abs(y[0])**n - 2*y[1]/xi])


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
    xi = np.array([.00001])
    switch = True
    while switch:
        F = f(xi[-1], [theta[-1], chi[-1]], n)
        theta = np.append(theta, theta[-1] + h*F[0])
        chi = np.append(chi, chi[-1] + h*F[1])
        xi = np.append(xi, xi[-1]+h)
        if theta[-1]*theta[-2]<0:
            switch = False
    return theta, xi, chi


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
    xi = np.array([.00001])
    switch = True
    while switch:
        F = RK4_step(xi[-1], [theta[-1], chi[-1]], f, h, n)
        theta = np.append(theta, F[0])
        chi = np.append(chi, F[1])
        xi = np.append(xi, xi[-1]+h)
        if theta[-1]*theta[-2]<0:
            switch = False
    return theta, xi , chi


y = np.array([1,0])



def error_n(xi0, xii, xiu, trinn):
    irele_err = np.array([1])
    urele_err = np.array([1])
    irelrk_err = np.array([1])
    urelrk_err = np.array([1])
    hi_trinn = xii/trinn
    hu_trinn = xiu/trinn
    j = hi_trinn
    i = hu_trinn
    while j != xii:
        print("euler irel", j)
        theta1, _, _ = euler (y, f, 3 / 2, j)
        print("rk4 irel", j)
        theta3, _, _ = RK4_method (y, f, 3 / 2, j)
        irele_err = np.append (irele_err, theta1[-1])
        irelrk_err = np.append (irelrk_err, theta3[-1])
        j += hi_trinn

    while i != xiu:
        print ("euler urel :", i)
        theta2,_,_ = euler (y, f, 3, i)
        print ("rk4 urel :", i)
        theta4,_,_ = RK4_method (y, f, 3, i)
        urele_err = np.append(urele_err, theta2[-1])
        urelrk_err = np.append(urelrk_err, theta4[-1])
        i += hu_trinn
    return irele_err[1:], urele_err[1:], irelrk_err[1:], urelrk_err[1:], hi_trinn, hu_trinn


#plott for oppg 3g

irele_err, urele_err, irelrk_err, urelrk_err, Ni, Nu = error_n(0.001, 3.653753736219229, 6.89684861937, 10)


plt.figure()
plt.plot(Ni, irele_err, '-.', label="irel euler error")

plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.ylabel("h error")
plt.xlabel("trinnlengde h")
plt.title("Irel error plott")
plt.show()
plt.figure()

plt.plot(Ni, irelrk_err, '-.', label="irel RK4 error")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.ylabel("h error")
plt.xlabel("trinnlengde h")
plt.title("Irel error plott")
plt.show()

plt.figure()
plt.plot(Nu, urele_err, '-.', label="urel euler error")
plt.plot(Nu, urelrk_err, '-.', label="urel RK4 error")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.title("Urel error plot")
plt.ylabel("h error")
plt.xlabel("trinnlengde h")
plt.show()

