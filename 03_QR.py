
import numpy as np

def inner_product(a,b):
  if(len(a) != len(b)):
    return
  return sum([a[i]*b[i] for i in range(len(a))])

def norm(a):
  return np.sqrt(sum([k**2 for k in a]))

def find_new_q(Q):
    n = Q.shape[0]
    random_vector = np.random.randn(n)
    v = random_vector.copy()
    for q in Q.T:
        v -= np.dot(q, random_vector) * q
    v = v / norm(v) # normalize
    v = np.array([v])
    return v

# Defining QR_decomposition and reduced_QR_Decomposition

def QR_decomposition(A):
  A = np.array(A)
  m,n = A.shape
  Q = []
  R = []
  for i in range(n):
    a_i = A[:,i]
    numerator = (a_i - sum([inner_product(a_i,Q[j])*Q[j] for j in range(i)]))
    denominator = norm(numerator)
    if denominator < 1e-12:
      raise ValueError("Matrix columns are linearly dependent")
    q_i = numerator/denominator
    r_val = [inner_product(a_i,Q[j]) for j in range(i)]
    r_val.append(denominator)
    for k in range(len(r_val),n):
      r_val.append(0)
    R.append(r_val)
    Q.append(q_i)
  Q = np.array(Q)
  R = np.array(R)
  Q = Q.T
  R = R.T
  print("---------------------FULL REPORT---------------------\n")
  print("Created matrix A :")
  print(A)
  print("_______________")
  print()
  print("REDUCED QR Decomposition : ")
  print("="*30)
  print()
  print("Matrix Q: ")
  print(Q)
  print()
  print("Matrix R: ")
  print(R)
  print()
  print("Reconstruction using Reduced QR :")
  print(Q@R)
  print()
  return Q,R

def QR_decomposition_full(A):
  Q,R = QR_decomposition(A)
  A = np.array(A)
  m,n = A.shape
  print()
  for j in range(n,m):
    q = find_new_q(Q)
    Q = np.hstack((Q,q.T))
    R = np.vstack((R,np.zeros((1,n))))
  print("FULL QR Decomposition : ")
  print("="*30)
  print()
  print("Matrix Q:")
  print(Q)
  print()
  print("Matrix R:")
  print(R)
  print()
  print("Reconstruction using Full QR :")
  print(Q@R)
  print()
  return Q,R

A = [       # To demonstrate 
    [1,1],
    [1,0],
    [0,1]
]


m = input("Row size : ")
n = input("Column size : ")
m,n = int(m), int(n)
A = np.zeros((m,n))
for i in range(m):
  for j in range(n):
    A[i,j] = int(input(f"Input value for A[{i+1}][{j+1}]: "))


A = np.array(A)

Q = QR_decomposition_full(A)
print("\n---------------------END REPORT---------------------")
