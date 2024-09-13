import random
import numpy as np
import pandas as pd

def generateVector(n,K):
  vector = [0]*n
  for i in range(0,K):
    idx = random.randint(0, n-1)
    vector[idx] = vector[idx] ^ 1
  return vector
def generateVectors(n,m,k):
  vectors = []
  for i in range(0,m):
    vectors.append(generateVector(n,k))
  return vectors

def checkLinearlyIndependent(vectors):
  mat = np.array(vectors)
  invertible, inv = invert(mat)
  if invertible:
    return True
  else:
    return False

def invert(matrix):
    n = len(matrix)
    inverse = np.identity(n)
    for col in range(n):
        pivot_row = None
        for row in range(col, n):
            if matrix[row][col] == 1:
                pivot_row = row
                break
        if pivot_row is not None:
            matrix[[col,pivot_row]] = matrix[[pivot_row,col]]
            inverse[[col,pivot_row]] = inverse[[pivot_row,col]]
            for row in range(n):
                if row == col:
                    continue
                if matrix[row][col] == 1:
                    matrix[row] = np.logical_xor(matrix[row], matrix[col])
                    inverse[row] = np.logical_xor(inverse[row], inverse[col])
    return (matrix.shape[0] == matrix.shape[1]) and (matrix == np.eye(matrix.shape[0])).all(), inverse
def checkProbability(n,m,k, iterations = 100):
  count = 0
  for i in range(0,iterations):
    vectors = generateVectors(n,m,k)
    if checkLinearlyIndependent(vectors):
      count = count + 1
  return count/iterations


#Fixed m,n
def getPlotsWithVaryingK(n,m, size_values_list, n_attempts):
  k_values = []
  p_values = []
  K = n//2

  if K%2 == 0:
    K = K + 1
  for i in size_values_list:
    prob = 0
    for j in range(0,n_attempts):
      k = int(i*n)
      if k%2 == 0:
         k = k+1
      prob += checkProbability(n,m,k)
    prob = prob/n_attempts
    print("n:",n,"m:",n,"k:",k," Probability:",prob)
    k_values.append(k)
    p_values.append(prob)
  return k_values, p_values

size_values_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
k_values_list = []
p_values_list = []
db_size_list = []






n = 128
m = 128
n_attempts = 1
k_values, p_values = getPlotsWithVaryingK(n,m,size_values_list, n_attempts)
db_size_values = [n for i in range(0,len(k_values))]
db_size_list.extend(db_size_values)
k_values_list.extend(k_values)
p_values_list.extend(p_values)


n = 256
m = 256
n_attempts = 1
k_values, p_values = getPlotsWithVaryingK(n,m,size_values_list, n_attempts)
db_size_values = [n for i in range(0,len(k_values))]
db_size_list.extend(db_size_values)
k_values_list.extend(k_values)
p_values_list.extend(p_values)



n = 512
m = 512
n_attempts = 1
k_values, p_values = getPlotsWithVaryingK(n,m,size_values_list, n_attempts)
db_size_values = [n for i in range(0,len(k_values))]
db_size_list.extend(db_size_values)
k_values_list.extend(k_values)
p_values_list.extend(p_values)

n = 1024
m = 1024
n_attempts = 1
k_values, p_values = getPlotsWithVaryingK(n,m,size_values_list, n_attempts)
db_size_values = [n for i in range(0,len(k_values))]
db_size_list.extend(db_size_values)
k_values_list.extend(k_values)
p_values_list.extend(p_values)


df = pd.DataFrame(zip(db_size_list, k_values_list, p_values_list), columns = [ 'db_size', 'k', 'Proabability'])
df.to_csv('bloom_swizzlers_exp4.csv')