import numpy as np
import matplotlib.pyplot as plt


def SPD_check(A):
  try:
      np.linalg.cholesky(A)
      return True
  except np.linalg.LinAlgError:
      return False

 
def conjugate_gradient(max_iter,A,b,tol,n):

  #Initialization

  x0 = np.zeros(n)
  r0 = b
  p0 = r0
 
  xk = x0
  rk = r0
  pk = p0
  points = [x0.copy()]

  for k in range(max_iter):
    if(np.linalg.norm(rk) < tol*np.linalg.norm(b)):
      break
    num = rk@rk
    den = pk@A@pk
    if abs(den) < 1e-12:
      print(f"Could not find alpha_{k}, due to small denominator error")
      break
    alpha_k = num/den

    xk = xk + (alpha_k*pk)

    r_prev = rk
    rk = rk - alpha_k*(A@pk)

    num = rk@rk
    den = r_prev@r_prev
    if abs(den) < 1e-12:
      print(f"Could not find beta_{k}, due to small denominator error")
      break
    beta_k = num/den
    pk = rk + beta_k*pk
    points.append(xk)


  x_final = xk
  print(A)
  return x_final,points



def get_matrix(m,n):
  A = np.zeros((m,n))
  for i in range(m):
    for j in range(n):
      A[i,j] = float(input(f"Input value for A[{i+1}][{j+1}]: "))
    print()
  return A

m = input("Row size : ") # Plotting only works for m = 2
m,n = int(m), int(m) # It only works for SPD matrices, So only square matrices
A = get_matrix(m,n)
"""
Use it when needed (Class example) !
A = [
    [2,0],
    [0,1/8]
    ]

A = np.array(A)
"""
print(A)


def executable(A):
  m,n = A.shape
  print("Enter b values b[1] and b[2] : ")
  b = []
  for i in range(m):
    print(f"b[{i+1}]: ")
    b_i = float(input(""))
    b.append(b_i)
  b = np.array(b)
  tol = float(input("Tolerance (e.g. 1e-6): "))
 
  max_iter = 100



  x_final,points = conjugate_gradient(max_iter,A,b,tol,n)
  points = np.array(points)


  print("Approximate solution : ",x_final)


  #plot iteration path
  plt.figure()
  plt.plot(points[:,0],points[:,1],'-o')

  for i in range(len(points)):
    plt.text(points[i,0], points[i,1] , f'{i}')

  plt.xlabel("X1")
  plt.ylabel("X2")
  plt.title("Path of the Conjugate Gradient algorithm")
  plt.grid(True)
  plt.show()

def __main__():
  if(not SPD_check(A)):
    print("Given matrix is not SPD, Cannot proceed further !")
  else :
    executable(A)

__main__()
