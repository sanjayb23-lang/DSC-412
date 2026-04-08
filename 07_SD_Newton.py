import numpy as np

# ================= USER FUNCTION =================
def f(x):
    return x[0]**2 + x[1]**2  # change this freely


# ================= NUMERICAL GRADIENT =================
def grad_f(x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy()
        x1[i] += h
        g[i] = (f(x1) - f(x)) / h
    return g


# ================= NUMERICAL HESSIAN =================
def hessian_f(x, h=1e-5):
    n = len(x)
    H = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            x1 = x.copy()
            x2 = x.copy()
            x3 = x.copy()
            x4 = x.copy()

            x1[i] += h; x1[j] += h
            x2[i] += h; x2[j] -= h
            x3[i] -= h; x3[j] += h
            x4[i] -= h; x4[j] -= h

            H[i,j] = (f(x1) - f(x2) - f(x3) + f(x4)) / (4*h*h)

    return H


# ================= PHI FUNCTIONS =================
def phi(alpha, xk, pk):
    return f(xk + alpha * pk)


def phi_prime(alpha, xk, pk):
    x = xk + alpha * pk
    return np.dot(grad_f(x), pk)


def phi_double_prime(alpha, xk, pk):
    x = xk + alpha * pk
    H = hessian_f(x)
    return pk @ H @ pk


# ================= FIND ALPHA =================
def find_alpha(pk, xk, tol=1e-6):
    alpha = 0.0

    for _ in range(100):
        num = phi_prime(alpha, xk, pk)
        den = phi_double_prime(alpha, xk, pk)

        if abs(den) < 1e-10:
            break

        alpha_new = alpha - num / den

        if abs(alpha_new - alpha) < tol:
            break

        alpha = alpha_new

    return alpha


# ================= STEEPEST DESCENT =================
def steepest_descent(x0, tol=1e-6, max_iter=100):

    x_old = np.array(x0, dtype=float)

    for i in range(max_iter):

        pk = -grad_f(x_old)

        alpha_k = find_alpha(pk, x_old)

        x = x_old + alpha_k * pk

        if np.linalg.norm(x - x_old) < tol:
            print("Converged in", i, "iterations")
            return x

        x_old = x

    return x_old


# ================= MAIN =================
x0 = [1, 1]

result = steepest_descent(x0)

print("Minimum at:", result)
