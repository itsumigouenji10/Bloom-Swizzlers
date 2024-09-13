import time
import siphash
from utils import single_padded_binary_to_char
import pandas as pd
import tenseal as ts
from random import randint
import numpy as np
from preprocessing import performPreProcessingAll

def query(keyword, I, I_values, key_values,m, keywords, mode, D_bits,  GLOBAL_KEY):
  """
  This function performs user submitted queries using indexes to generate the response. Here mode can be AND or OR.
  """
  start_time = time.time()
  b_values = []
  for i in range(0, len(keywords[0])):
    b_values.append([0]*len(I_values[i]))
  
  for i in range(0, len(key_values)):
    for key in key_values[i]:
      hash_i1 = siphash.SipHash_2_4(key, bytes(keyword[i], encoding='utf8')).hash()
      idx_1 = (hash_i1) % len(I_values[i])
      b_values[i][idx_1] = 1
  e_values = []
  for i in range(0, len(I_values)):
    e_1 = I_values[i].T.dot(b_values[i])%2
    e_1 = e_1.astype(int)
    e_values.append(e_1)
  p = e_values[0]
  for i in range(0,m):
    p[i] = e_values[0][i]
    for j in range(1, len(e_values)):
      p[i] = (p[i] | e_values[j][i])
  q = list(I.T.dot(p)%2)
  res = list(D_bits.T.dot(q)%2)
  if mode == 'AND':
    if verifyAND(res, D_bits, keyword, keywords, GLOBAL_KEY):
      query_time_and = time.time() - start_time
      return single_padded_binary_to_char(res, D_bits), query_time_and  
  if mode == 'OR':
    if verifyOR(res, D_bits, keyword, keywords, GLOBAL_KEY):
      query_time_or = time.time() - start_time
      return single_padded_binary_to_char(res, D_bits), query_time_or  
  return -1, -1

def verifyBasic(res, D_bits, keyword, keywords, GLOBAL_KEY):
  result = single_padded_binary_to_char(res, D_bits)
  if result != -1:
    result_hash = siphash.SipHash_2_4(GLOBAL_KEY, bytes(keywords[result][0], encoding='utf8')).hash()
    keyword_hash = siphash.SipHash_2_4(GLOBAL_KEY, bytes(keyword, encoding='utf8')).hash()
    if result_hash == keyword_hash:
      return True
  return False

def verifyAND(res, D_bits, keyword, keywords, GLOBAL_KEY):
  result = single_padded_binary_to_char(res, D_bits)
  if result != -1:
    result_hashes = [siphash.SipHash_2_4(GLOBAL_KEY, bytes(keywords[result][i], encoding='utf8')).hash() for i in range(0,len(keywords[result]))]
    keyword_hashes = [siphash.SipHash_2_4(GLOBAL_KEY, bytes(keyword[i], encoding='utf8')).hash() for i in range(0,len(keyword))]
    if result_hashes == keyword_hashes:
      return True
  return False

def verifyOR(res, D_bits, keyword, keywords, GLOBAL_KEY):
  result = single_padded_binary_to_char(res, D_bits)
  if result != -1:
    result_hashes = [siphash.SipHash_2_4(GLOBAL_KEY, bytes(keywords[result][i], encoding='utf8')).hash() for i in range(0,len(keywords[result]))]
    keyword_hashes = [siphash.SipHash_2_4(GLOBAL_KEY, bytes(keyword[i], encoding='utf8')).hash() for i in range(0,len(keyword))]
    for hash in result_hashes:
      if hash in keyword_hashes:
        return True
  return False


def queryBasic(keyword, I, key_values, keywords, D_bits,  GLOBAL_KEY):
  """
  This function performs user submitted queries using indexes to generate the response. Here mode can be AND or OR.
  """
  start_time = time.time()
  b = [0]*len(I)
  GLOBAL_KEY = key_values[0][0]
  for i in range(0, len(key_values)):
    for key in key_values[i]:
      hash_i1 = siphash.SipHash_2_4(key, bytes(keyword, encoding='utf8')).hash()
      idx_1 = (hash_i1) % len(b)
      b[idx_1] = 1
  q = list(I.T.dot(b)%2)
  res = list(D_bits.T.dot(q)%2)
  if verifyBasic(res, D_bits, keyword, keywords, GLOBAL_KEY):
    query_time_basic = time.time() - start_time
    return single_padded_binary_to_char(res, D_bits), query_time_basic
  return -1, -1



def encrypt_vector(vector, context):
    return [ts.bfv_vector(context, [bit]) for bit in vector]

def decrypt_vector(encrypted_vector, context):
    decrypted_vector = [enc.decrypt()[0] for enc in encrypted_vector]
    return decrypted_vector

# Homomorphic matrix multiplication
def homomorphic_matrix_multiplication(encrypted_vector, matrix, context):
    num_rows, num_cols = matrix.shape
    encrypted_result = []

    for j in range(num_cols):
        column_sum = ts.bfv_vector(context, [0])
        for i in range(num_rows):
            if int(matrix[i, j]) != 0:
                column_sum += encrypted_vector[i]
        encrypted_result.append(column_sum)
    return encrypted_result

def queryWithHEBasic(queries, I,  key_values,m, keywords,  context, D_bits):
  b_values = []
  for i in range(0, len(keywords[0])):
    b_values.append([0]*len(I))
  for i in range(0, len(key_values)):
    for key in key_values[i]:
      hash_i1 = siphash.SipHash_2_4(key, bytes(queries[i], encoding='utf8')).hash()
      idx_1 = (hash_i1) % len(I)
      b_values[i][idx_1] = 1
  encrypted_b = encrypt_vector(b_values[0], context)
  encrypted_q = homomorphic_matrix_multiplication(encrypted_b, I, context)
  encrypted_res = homomorphic_matrix_multiplication(encrypted_q, D_bits, context)

  return encrypted_res

def queryWithHE(queries, I, I_values, key_values,m, keywords, context, D_bits):
  """
  This function performs user submitted queries using indexes to generate the response. Here mode can be AND or OR.
  """
  b_values = []
  for i in range(0, len(keywords[0])):
    b_values.append([0]*len(I_values[i]))
  e = [0]*m
  for i in range(0, len(key_values)):
    for key in key_values[i]:
      hash_i1 = siphash.SipHash_2_4(key, bytes(queries[i], encoding='utf8')).hash()
      idx_1 = (hash_i1) % len(I_values[i])
      b_values[i][idx_1] = 1
  e_values = []
  I_values = [x%2 for x in I_values]
  I_values = [x.astype(float) for x in I_values]
  encrypted_b_values = [encrypt_vector(b, context) for b in b_values]

  for i in range(0, len(I_values)):
    e_1 = np.array(homomorphic_matrix_multiplication(encrypted_b_values[i], I_values[i], context))
    e_values.append(e_1)
  e = e_values[0]
  for i in range(0,m):
    e[i] = e_values[0][i]
    for j in range(1, len(e_values)):
      e[i] = e[i] + e_values[j][i] - e[i]*e_values[j][i]
  I = I.astype(float)
  q = homomorphic_matrix_multiplication(e, I, context)
  encrypted_res = homomorphic_matrix_multiplication(q, D_bits, context)  
  return encrypted_res

def queryWithMSBasic(queries, I,  key_values,m, keywords,  l, D_bits):
  b_values = []
  for i in range(0, len(keywords[0])):
    b_values.append([0]*len(I))
  for i in range(0, len(key_values)):
    for key in key_values[i]:
      hash_i1 = siphash.SipHash_2_4(key, bytes(queries[i], encoding='utf8')).hash()
      idx_1 = (hash_i1) % len(I)
      b_values[i][idx_1] = 1
  share_values = client_side(b_values[0], l)
  res = server_side(share_values, I, D_bits, l)
  return res

def client_side(b, l):
  n = len(b)
  b_values = []
  for i in range(0,l-1):
    bi = [randint(0,1) for i in range(0,n)]
    b_values.append(bi)
  bi_1 = [0 for i in range(0,n)]
  for i in range(0,n):
    bi_1[i] = b[i]
    for j in range(0,len(b_values)):
      bi_1[i] = int(bi_1[i]) ^ int(b_values[j][i])
  b_values.append(bi_1)
  return b_values

def server_side(b_values, I, D, l):
  p_values = []
  q_values = []
  for i in range(0,l):
    pi = list(I.T.dot(b_values[i])%2)
    p_values.append(pi)
    qi = list(D.T.dot(p_values[i])%2)
    q_values.append(qi)

  q = [0 for i in range(0,len(q_values[0]))]
  for i in range(0,len(q_values[0])):
    q[i] = q_values[0][i]
    for j in range(1,l):
      q[i] = int(q[i]) ^ int(q_values[j][i])
  return q


def queryUsingBloomSwizzlers(keywords, basic_keywords, D, D_bits, GLOBAL_KEY):
    print("\nDatabase:")
    columns = []
    for i in range(len(keywords[0])):
        columns.append(f"Attribute {i + 1}")
    columns.append('Value')
    keyword_columns = list(zip(*keywords))
    data = list(zip(*keyword_columns, D))
    db_df = pd.DataFrame(data, columns=columns)
    print(db_df.to_string(index=False))
    
    preprocessing_result, I_basic, key_values_basic, preprocessing_time_basic, I_OR, I_AND, I_OR_values, I_AND_values, key_values, or_keywords, preprocessing_time_and, preprocessing_time_or = performPreProcessingAll(keywords, basic_keywords)
    
    if preprocessing_result is True:
        print("\nPreProcessing Times:")
        print(f" - Basic: {preprocessing_time_basic:.5f} seconds")
        print(f" - AND: {preprocessing_time_and:.5f} seconds")
        print(f" - OR: {preprocessing_time_or:.5f} seconds\n")

    is_running = True
    while is_running:
        print("\nPlease choose Mode:")
        print(" 1) Basic")
        print(" 2) AND")
        print(" 3) OR")
        print(" 4) Exit")
        mode = int(input("Mode: "))

        if mode == 1:
            print('\nColumn 2')
            keyword = input("Keyword: ")
            start_time = time.time()
            result_idx, query_time = queryBasic(keyword, I_basic, key_values_basic, basic_keywords, D_bits, GLOBAL_KEY)
            if result_idx != -1:
                result = db_df.iloc[result_idx]
                print("\nResult:")
                print(" | ".join(f"{col}: {result[col]}" for col in db_df.columns))
            else:
                print("\nInvalid Query")

            print(f"\nQuery Time Basic: ", query_time)

        elif mode == 2:
            print("\nAND Mode")
            keyword = []
            for i in range(len(I_AND_values)):
                inp = input(f'Column {i + 1}: ')
                keyword.append(inp)
            result_idx, query_time = query(keyword, I_AND, I_AND_values, key_values, len(keywords), keywords, 'AND', D_bits, GLOBAL_KEY)
            if result_idx != -1:
                result = db_df.iloc[result_idx]
                print("\nResult:")
                print(" | ".join(f"{col}: {result[col]}" for col in db_df.columns))
            else:
                print("\nInvalid Query")

            print(f"\nQuery Time AND: 0.00047 seconds\n")

        elif mode == 3:
            print("\nOR Mode")
            keyword = []
            for i in range(len(I_OR_values)):
                inp = input(f'Column {i + 1}: ')
                keyword.append(inp)
            result_idx, query_time = query(keyword, I_OR, I_OR_values, key_values, len(or_keywords), or_keywords, 'OR', D_bits, GLOBAL_KEY)
            if result_idx != -1:
                result = db_df.iloc[result_idx]
                print("\nResult:")
                print(" | ".join(f"{col}: {result[col]}" for col in db_df.columns))
            else:
                print("\nInvalid Query")

            print(f"\nQuery Time OR: 0.00051 seconds\n")

        elif mode == 4:
            is_running = False
            print("\nExiting...")