import matplotlib.pyplot as plt
import numpy as np

def f(xi, y, n): return np.array([y[1], -np.abs(y[0])**(n) - 2*y[1]/xi])


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
    L=0
    while switch:
        L+=1
        print("L:", L)
        F = RK4_step(xi[-1], [theta[-1], chi[-1]], f, h, n)
        theta = np.append(theta, F[0])
        chi = np.append(chi, F[1])
        xi = np.append(xi, xi[-1]+h)
        if theta[-1]*theta[-2]<0:
            switch = False
    return theta, xi , chi


y = np.array([1,0])
#n = 1
theta, xi, _ = RK4_method(y, f, 3/2, 0.9)

def error_n(h0, h1, trinn):
    """
    theta1 ,_,_ = euler (y, g1, g2, h0, 3 / 2)
    theta2 ,_,_= euler (y, g1, g2, h0, 3)
    theta3 ,_,_= RK4_method(y, f, 3/2, h0)
    theta4 ,_,_= RK4_method (y, f, 3, h0)
    """
    irele_err = np.array([1])
    urele_err = np.array([1])
    irelrk_err = np.array([1])
    urelrk_err = np.array([1])
    h_list = np.linspace(h1, h0, trinn)
    for i in h_list:
        print ("i :", i)
        theta1,_ ,_ = euler (y, f, 3/2, i)
        print ("i :", i)
        theta2,_,_ = euler (y, f, 3, i)
        print ("i :", i)
        theta3,_,_ = RK4_method (y, f, 3 / 2, i)
        print ("i :", i)
        theta4,_,_ = RK4_method (y, f, 3, i)
        irele_err = np.append(irele_err, theta1[-1])
        urele_err = np.append(urele_err, theta2[-1])
        irelrk_err = np.append(irelrk_err, theta3[-1])
        urelrk_err = np.append(urelrk_err, theta4[-1])
        print("i :",i)
    return irele_err, urele_err, irelrk_err, urelrk_err


#plott for oppg 3g

irele_err, urele_err, irelrk_err, urelrk_err = error_n(0.001, 0.9, 2)

x = np.linspace(0.001, 0.9, 2)

plt.figure()
plt.plot(x, irele_err, '-.', label="irel euler error")
plt.plot(x, irelrk_err, '-.', label="irel RK4 error")
plt.legend()
plt.ylabel("h error")
plt.xlabel("trinnlengde h")
plt.show()

plt.figure()
plt.plot(x, urele_err, '-.', label="urel euler error")
plt.plot(x, urelrk_err, '-.', label="urel RK4 error")
plt.legend()
plt.ylabel("h error")
plt.xlabel("trinnlengde h")
plt.show()

