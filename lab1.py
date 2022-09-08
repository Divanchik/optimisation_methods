def f1(x):
    return 2*x**3 + x**2

def dichotomy(f, a, b, eps=0.01, delta=0.001):
    counter = 0
    while b - a > 2 * eps:
        counter += 1
        x1 = (a+b) / 2 - delta
        x2 = (a+b) / 2 + delta
        if f(x1) > f(x2):
            a = x1
        else:
            b = x2
    return (a+b)/2, counter

def golden_ratio(f, a, b, eps=0.01):
    x1s = 0
    x2s = 0
    counter = 0
    while b - a > 2 * eps:
        counter += 1
        x1 = b - (b - a) * 0.618 if x1s == 0 else x1s
        x2 = a + (b - a) * 0.618 if x2s == 0 else x2s
        if f(x1) > f(x2):
            x1s = x2
            x2s = 0
            a = x1
        else:
            x1s = 0
            x2s = x1
            b = x2
    return (a+b)/2, counter

def fibbonachi(f, a, b, N):
    counter = 0

print(dichotomy(f1, -0.4, 0.4))
print(golden_ratio(f1, -0.4, 0.4))
