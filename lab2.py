from scipy.optimize import minimize_scalar
import numpy as np
from math import pi, tanh, cosh, sin, cos
import matplotlib.pyplot as plt


def rotate(x = None, y = None):
    x1 = (x-3)*cos(pi/6) + (y-1)*sin(pi/6)
    y1 = (y-1)*cos(pi/6) - (x-3)*sin(pi/6)
    return x1, y1


def f1(x, y, z):
    x1, y1 = rotate(x, y)
    return tanh(x1)**2 + y1**4 + (z-3)**2


def f2(a):
    return f1(a[0], a[1], a[2])


# 2tanh(x')/cosh(x')^2
# 4y'^3
# 2z - 6
def gradient(x, y, z):
    x1, y1 = rotate(x, y)
    tmp1 = 2 * tanh(x1) / (cosh(x1)*cosh(x1))
    tmp2 = 4*y1**3
    dx = tmp1 * cos(pi/6) - tmp2 * sin(pi/6)
    dy = tmp1 * sin(pi/6) + tmp2 * cos(pi/6)
    dz = 2 * z - 6
    return np.array([dx, dy, dz], float)


def H(a):
    phi = pi/6
    x1, y1 = rotate(a[0], a[1])
    txx = -2 * cos(phi) * (cosh(2*x1)-2) / cosh(x1)**4
    txy = -12 * y1**2 * sin(phi)
    tyx = -2 * sin(phi) * (cosh(2*x1)-2) / cosh(x1)**4
    tyy = 12 * y1**2 * cos(phi)
    dxx = txx * cos(phi) - txy * sin(phi)
    dxy = txx * sin(phi) + txy * cos(phi)
    dyx = tyx * cos(phi) - tyy * sin(phi)
    dyy = tyx * sin(phi) + tyy * cos(phi)
    h = [
        [dxx, dxy, 0],
        [dyx, dyy, 0],
        [0, 0, 2]
    ]
    return np.array(h, float)


def coords_method(f, a, eps=0.0001, log=False):
    x0, y0, z0 = a[0], a[1], a[2]
    while True:
        x1 = minimize_scalar(lambda arg: f(arg, y0, z0)).x
        y1 = minimize_scalar(lambda arg: f(x1, arg, z0)).x
        z1 = minimize_scalar(lambda arg: f(x1, y1, arg)).x
        if x1-x0 < eps and y1-y0 < eps and z1-z0 < eps:
            x0, y0, z0 = x1, y1, z1
            break
        x0, y0, z0 = x1, y1, z1
    return np.array([x0, y0, z0], float).round(1)


def gradient_method(f, a, eps=0.0001, log=False):
    while True:
        if log:
            print(a)
        speed = minimize_scalar(lambda arg: f(
            a - arg * gradient(a[0], a[1], a[2]))).x
        a1 = a - speed * gradient(a[0], a[1], a[2])
        if np.linalg.norm(a1-a) < eps:
            return a1.round(1)
        a = np.copy(a1)


def newton_method(a: np.ndarray, eps=0.0001, log=False):
    xk = a.copy()
    while True:
        g = gradient(xk[0], xk[1], xk[2])
        if log: print(xk.round(1), -g.round(1))
        h = H(xk)
        if np.linalg.det(h) == 0:
            raise ValueError('H matrix determinant is zero!', h)
        ih = np.linalg.inv(h)
        xk1 = xk - ih @ g
        # if log: print(xk1.round(6), -g.round(6))
        if np.linalg.norm(g) < eps:
            return xk1.round(1)
        xk = np.copy(xk1)


a = np.array([1, -2, 0], float)
print(coords_method(f1, a))
print(gradient_method(f2, a))
print(newton_method(a, log=True))
print("Answer:", [3, 1, 3])
# (3, 1, 3)