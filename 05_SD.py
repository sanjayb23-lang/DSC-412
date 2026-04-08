import numpy as np
import matplotlib.pyplot as plt

def steepest_descent(A, b, x0, tol, max_iter=1000):

    x = np.array(x0, dtype=float)
    path = [x.copy()]

    for k in range(max_iter):

     
        pk = b - A @ x

       
        if np.linalg.norm(pk) < tol:
            print(f"Converged in {k} iterations")
            break

       
        alpha = (pk @ pk) / (pk @ (A @ pk))


        x = x + alpha * pk
        path.append(x.copy())

    return x, np.array(path)
    
    

# ===== USER INPUT =====

n = int(input("Enter dimension (n): "))

print("Enter matrix A row-wise:")
A = []
for i in range(n):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    A.append(row)

A = np.array(A)

print("Enter vector b:")
b = np.array(list(map(float, input().split())))

print("Enter initial point x0:")
x0 = np.array(list(map(float, input().split())))

tol = float(input("Enter tolerance: "))


# ===== RUN =====

minimum, path = steepest_descent(A, b, x0, tol)

print("Minimum:", minimum)


# ===== PLOT (only for 2D) =====

x_vals = np.linspace(-6,6,100)
y_vals = np.linspace(-6,6,100)

X,Y = np.meshgrid(x_vals,y_vals)

Z = 0.5*(A[0,0]*X**2 + 2*A[0,1]*X*Y + A[1,1]*Y**2) - (b[0]*X + b[1]*Y)

plt.contour(X,Y,Z,50)
plt.plot(path[:,0], path[:,1], 'ro-')

plt.title("Steepest Descent Path")
plt.xlabel("x1")
plt.ylabel("x2")

plt.show()
