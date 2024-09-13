import numpy as np
from utils import invert, generateKeys
import siphash
import time
import os
import itertools

def create_inverse_B(Keywords, m, keys):
  """
  This function is used ro create Bloom Filters Matrix (B) for a set of Keywords (1 column) and Keys.
  It tries to create LI vectors for Bloom Filters Matrix (B) for given set of Keywords. Here, m = length of view.
  """
  number_of_keywords = len(Keywords)
  n = number_of_keywords
  k = m//2
  if k%2 == 0:
    k += 1
  bloom_filters = []
  for keyword in Keywords:
    vector = [0]*m
    for key in keys:
      hash_i1 = siphash.SipHash_2_4(key, bytes(keyword, encoding='utf8')).hash()
      idx_1 = (hash_i1) % m
      vector[idx_1] = 1
    bloom_filters.append(vector)
  bloom_filters_copy = bloom_filters.copy()
  B = np.array(bloom_filters_copy)
  invertible, B_inv = invert(np.array(bloom_filters_copy))
  B_inv = B_inv.astype(int)
  return invertible, B,B_inv,keys

def generateB(keywords,R):
  """
  This function generates Bloom Filters for each set of keywords (multiple columns) given a value of 
  R (number of randomly generated bits in keys). It uses the create_inverse_B(Keywords,m, keys) to generate a B matrix 
  for each column.
  """
  columns = []
  for i in range(0,len(keywords[0])):
    column = []
    for j in range(0, len(keywords)):
      if keywords[j][i] not in column:
        column.append(keywords[j][i])
      if column not in columns:
        columns.append(column)
  m_values = [len(i) for i in columns]
  #k = int(np.log2(min(m_values)))
  k = min(m_values)//2
  if k%2 == 1:
    k = k+1
  attempts = 0
  while(True):
    B_values = []
    B_inv_values = []
    key_values = []
    attempts += 1
    random_keys = []
    for i in range(0,k):
      random_keys.append(os.urandom(R))
    keys = generateKeys(k, 1, random_keys)
    for i in range(0,len(m_values)):
      inv, B, B_inv, keys_value = create_inverse_B(columns[i], m_values[i], keys)
      if inv:
        B_values.append(B)
        B_inv_values.append(B_inv)
        key_values.append(keys_value)
        continue
      else:
        break
    if len(B_values) == len(m_values):
      return True, [B_values, B_inv_values, key_values], attempts
    else:
      if attempts > 1000:
        return False, "Stopped after 1000 attempts", attempts
      continue

def create_inverse_M(Keywords, m, k):
  """
  This function generates LI vectors to create M and it's inverse for a set of Keywords, given values of m and k.
  """
  #m is the number of rows that are retrievable so I guess the total number of keypairs in C
  number_of_keywords = len(Keywords)

  n = number_of_keywords
  #k = int(np.log2(m))
  if k%2 == 0:
    k += 1
  attempts = 0
  c = 0
  while(True):
    c = c+1
    vector_values = []
    M_values = []
    for i in range(0,len(Keywords[0])):
      vector_values.append([])
      M_values.append([])
    M = []
    siphashes = []
    keys = []
    for i in range(0,k):
      random_key = os.urandom(16)
      keys.append(random_key)
    for keyword in Keywords:
      for i in range(0,len(Keywords[0])):
        vector_values[i] = [0]*m
      vector = [0]*m
      for i in range(0, len(Keywords[0])):
        for key in keys:
            hash_i1 = siphash.SipHash_2_4(key, bytes(keyword[i], encoding='utf8')).hash()
            idx_1 = (hash_i1) % m
            vector_values[i][idx_1] = 1
        siphashes.append((keyword[i]))
      for i in range(0,m):
        vector[i] = vector_values[0][i]
        for j in range(1,len(Keywords[0])):
          vector[i] = vector[i] | vector_values[j][i]
      for i in range(0, len(Keywords[0])):
        if vector_values[i] not in M_values[i]:
          M_values[i].append(vector_values[i])
      M.append(vector)
    attempts = attempts + 1
    M_copy = M.copy()
    M_final = np.array(M_copy)
    invertible, M_inv = invert(np.array(M_copy))
    if invertible:
      #print("Found suitable system after " + str(attempts) + " attempts")
      return True, [M_final,M_inv,keys, M_values], attempts
    else:
      if attempts > 10000:
        return False, 'Unable to find LI vectors after'+str(attempts)+'attempts', attempts

def getM(keywords, mode):
  """
  This function generates an M matrix while also noting down the time for generating it. 
  It uses the create_inverse_M(keywords,m,k) to generate the matrix.
  """
  n = len(keywords)
  k = len(keywords)//2
  flag, result, attempts = create_inverse_M(keywords, len(keywords),k)
  if not flag:
     print(result)
  return flag, result, attempts

def getB(keywords, R, mode):
  """
  This function generates a B matrix while also noting down the time for generating it. 
  It uses the create_inverse_B(keywords,R) to generate the matrix.
  """
  # Step 2: Generate B1 and B2 using same set of keys with R random bits
  start_time = time.time()
  flag, result, attempts = generateB(keywords,R)
  return flag, result, attempts

def performPreProcessing(keywords,  R=8):
  """
  This function performs the Pre-Porcessing, which is generation of B, M and I matrices. 
  """
  start_time = time.time()
  keyword_values = []
  for j in range(0,len(keywords[0])):
   column = []
   for i in range(0,len(keywords)):
    if keywords[i][j] not in column:
      column.append(keywords[i][j])
   keyword_values.append(column)
  and_keywords = list(itertools.product(*keyword_values))
  flag,result, and_attempts = getM(and_keywords, 'AND')
  if flag == True:
    M,M_inv, keys, M_values = result[0], result[1], result[2], result[3]
    flag, result, _ = getB(keywords, R, 'OR')
    if flag == True:
      B_values = result[0]
      B_inv_values = result[1]
      key_values = result[2]

      Q_AND = []
      row_size = len(keywords)
      for keyword in and_keywords:
        indexes = [0 for i in range(0,row_size)]
        for i in range(0,len(keywords)):
         flag = 1
         for j in range(0, len(keywords[0])):
            if keywords[i][j] != keyword[j]:
             flag = 0
         if flag == 1:
            indexes[i] = 1
            break
        Q_AND.append(indexes)
      I_AND = np.matmul(M_inv,Q_AND)
      I_AND_values = []
      for i in range(0, len(keywords[0])):
        I = np.matmul(B_inv_values[i], M_values[i])
        I_AND_values.append(I)
      preprocessing_time_and = time.time() - start_time
      start_time = time.time()
      or_keywords = list(itertools.product(*keyword_values))
      flag,result, or_attempts = getM(or_keywords, 'OR')
      if flag == True:
        M,M_inv, keys, M_values = result[0], result[1], result[2], result[3]
        Q_OR = []
        row_size = len(keywords)
        for keyword in or_keywords:
          indexes = [0 for i in range(0,row_size)]
          for i in range(0,len(keywords)):
            flag = 0
            for j in range(0, len(keywords[0])):
              if keywords[i][j] == keyword[j]:
                flag = 1
            if flag == 1:
              indexes[i] = 1
              break
          Q_OR.append(indexes)

        I_OR = np.matmul(M_inv,Q_OR)
        I_OR_values = []
        for i in range(0, len(or_keywords[0])):
          I = np.matmul(B_inv_values[i], M_values[i])
          

          I_OR_values.append(I)
        preprocessing_time_or = time.time() - start_time
        return True, [I_OR, I_AND,I_OR_values, I_AND_values, key_values, or_keywords], preprocessing_time_and, preprocessing_time_or, and_attempts, or_attempts
      else:
        return False, result, 0, 0, and_attempts, or_attempts
    else:
      return False, result, 0, 0, and_attempts, or_attempts
  else:
    return False, result, 0, 0, and_attempts, or_attempts

def performPreProcessingBasic(keywords,  R=8):
  """
  This function performs Pre-Processing for the Basic version of Bloom Swizzlers. (B*I = Q).
  It uses getB() function to create the B matrix.
  It uses the getM() function to create the I (M) matrix.
  """
  start_time = time.time()
  flag, result, basic_attempts = getB(keywords, R, 'Basic')
  if flag == True:
    B_values = result[0]
    B_inv_values = result[1]
    key_values = result[2]
    Q_AND = np.eye(len(keywords))
    I_AND = np.matmul(B_inv_values[0],Q_AND)
    I_values = []
    for i in range(0, len(keywords[0])):
      I = np.matmul(B_inv_values[i], B_values[i])
      I_values.append(I)
    pre_processing_time_basic = time.time() - start_time
    return True, [I_AND,I_values, key_values], pre_processing_time_basic, basic_attempts
  else:
    return False, result, -1, basic_attempts

def performPreProcessingAll(keywords, basic_keywords):
  flag, result, preprocessing_time_basic, basic_attempts  = performPreProcessingBasic(basic_keywords)
  if flag is True:
    I_basic,_, key_values_basic = result[0], result[1],result[2]
    flag, result, preprocessing_time_and, preprocessing_time_or, and_attempts, or_attempts = performPreProcessing(keywords)
    if flag is True:
      I_OR, I_AND,I_OR_values, I_AND_values, key_values, or_keywords = result[0], result[1],result[2], result[3], result[4], result[5]
      return True, I_basic, key_values_basic, preprocessing_time_basic, I_OR, I_AND,I_OR_values, I_AND_values, key_values, or_keywords, preprocessing_time_and, preprocessing_time_or
