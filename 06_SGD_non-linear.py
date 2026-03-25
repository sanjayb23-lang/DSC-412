import numpy as np
def f(x):
    return (x[0])**2 + (x[1])**4

tol = 1e-8
 
def grad_f(x):
    return np.array([2*x[0],4*x[1]**3])

def findalpha(pk,xk):

    def phi(alpha):
        return f(xk + alpha * P)

    def phi_prime(alpha):
        return np.dot(grad_f(xk + alpha * P), P)

    def phi_double_prime(alpha):
        h = 1e-5
        return (phi_prime(alpha + h) - phi_prime(alpha - h)) / (2*h)
    
    y_old = 0.0
    y = 0.0

    for i in range(100):
        denom = phi_double_prime(y_old)
        if abs(denom) < 1e-12:
            break

        y = y_old - phi_prime(y_old) / denom

        # stopping condition
        if abs(y - y_old) / max(1, abs(y_old)) < tol:
            break

        y_old = y

    return y
x_old = np.array([20.0, 1.0])
x = np.array([1.0, 1.0])

MaxIter = 100

for i in range(MaxIter):

    P = -grad_f(x_old)

    alpha = findalpha(P, x_old)

    x = x_old + alpha * P

    # stopping condition
    if np.linalg.norm(x - x_old) / max(1, np.linalg.norm(x_old)) < tol:
        break

    x_old = x


print("The minimizer =", x)
print("f(x) =", f(x))
