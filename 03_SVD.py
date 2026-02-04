import numpy as np

# Input matrix
m = int(input("Enter number of rows: "))
n = int(input("Enter number of columns: "))

print("Enter matrix row by row:")
A = np.array([list(map(float, input().split())) for _ in range(m)])

# Step 1: A^T A
ATA = A.T @ A

# Step 2: Eigen-decomposition
eigvals, eigvecs = np.linalg.eigh(ATA)

# Step 3: Sort descending
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Step 4: Singular values (handle numerical noise)
eigvals[eigvals < 0] = 0
singular_vals = np.sqrt(eigvals)

V = eigvecs

# Step 5: Compute U
U = np.zeros((m, m))
r = np.sum(singular_vals > 1e-12)

for i in range(r):
    U[:, i] = (A @ V[:, i]) / singular_vals[i]

# Gram-Schmidt to complete U
def gram_schmidt(V):
    Q = []
    for v in V.T:
        for q in Q:
            v = v - np.dot(q, v) * q
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            Q.append(v / norm)
    return np.array(Q).T

if r < m:
    candidates = np.column_stack((U[:, :r], np.eye(m)))
    U = gram_schmidt(candidates)

# Step 6: Sigma matrix
Sigma = np.zeros((m, n))
for i in range(r):
    Sigma[i, i] = singular_vals[i]

# Reconstruction
A_reconstructed = U @ Sigma @ V.T

print("\nOriginal A:\n", A)
print("sigma :",Sigma)
print("V-T:",V.T)
print("\nReconstructed A:\n", A_reconstructed)
print("\nReconstruction Error:", np.linalg.norm(A - A_reconstructed))

