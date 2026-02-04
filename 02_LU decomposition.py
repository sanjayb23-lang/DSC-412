import numpy as np

n = int(input("Enter matrix size (n x n): "))

print("Enter matrix row by row:")
A = np.array([list(map(float, input().split())) for _ in range(n)])

L = np.zeros((n, n))
U = np.zeros((n, n))

for i in range(n):
    L[i, i] = 1

    # Compute U
    for j in range(i, n):
        U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))

    if abs(U[i, i]) < 1e-12:
        raise ValueError("Zero or near-zero pivot encountered. LU without pivoting not possible.")

    # Compute L
    for j in range(i + 1, n):
        L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

print("\nL matrix:\n", L)
print("\nU matrix:\n", U)
print("\nCheck A = L @ U :", np.allclose(A, L @ U))

