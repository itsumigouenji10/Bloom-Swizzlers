import tenseal as ts
import time
import pandas as pd
import numpy as np
from preprocessing import performPreProcessingBasic
from querying import queryWithHEBasic, decrypt_vector, queryWithMSBasic

def performBloomSwizzlersBasic(keywords, D_bits, idx, n_attempts, R=8):
  """
  This function creates an interface in Python Terminal, for performing AND and OR Bloom Swizzlers queries.
  """
  #print(keywords)
  flag, result, preprocessing_time_basic, attempts = performPreProcessingBasic(keywords,R)
  if flag is True:
    I, I_values, key_values = result[0], result[1],result[2]
    queries = keywords[0]
    start_time = time.time()
    context = ts.context(
    ts.SCHEME_TYPE.BFV,
    poly_modulus_degree=8192,
    plain_modulus=1032193  # A prime number greater than your maximum value
)
    context.generate_galois_keys()
    context.generate_relin_keys()
    key_gen_time = time.time() - start_time
    query_time_he = 0
    for j in range(0,n_attempts):
      start_time = time.time()
      encrypted_res = queryWithHEBasic(keywords[idx], I,  key_values,len(keywords), keywords,  context,  D_bits)
      decrypted_res = [int(x)%2 for x in  decrypt_vector(encrypted_res, context)]
      res = single_padded_binary_to_char(decrypted_res, D_bits, keywords)
      query_time_he += time.time() - start_time
    query_time_he = query_time_he/n_attempts
    l = 4
    query_time_ms = 0
    for j in range(0,n_attempts):
      start_time = time.time()
      res_arr = queryWithMSBasic(keywords[0], I,  key_values,len(keywords), keywords,  l, D_bits)
      res = single_padded_binary_to_char(res_arr, D_bits, keywords)
      query_time_ms += time.time() - start_time
    query_time_ms = query_time_ms/n_attempts
    return query_time_he, query_time_ms


def chars_to_padded_binary(D):
    # Convert characters to their binary representation
    D_bits = [format(int(x), 'b') if x.isdigit() else format(ord(x), 'b') for x in D]

    # Find the length of the longest binary string
    max_len = max(len(bits) for bits in D_bits)

    # Pad all binary strings to the length of the longest one
    D_bits_padded = [bits.zfill(max_len) for bits in D_bits]

    # Convert strings to lists of integers
    D_bits_padded = [[int(bit) for bit in bits] for bits in D_bits_padded]

    # Convert the list of lists to a numpy array
    D_bits_array = np.array(D_bits_padded)

    return D_bits_array


def single_padded_binary_to_char(D_bits_single, D_bits, keywords):
  if D_bits_single in D_bits:
    return keywords[D_bits.tolist().index(D_bits_single)]
  else:
    -1

db_size_range = [2**i for i in range(4, 13)]
size_list = []
query_time_he_list = []
query_time_ms_list = []
n_attempts = 3

for db_size in db_size_range:
  n = db_size
  D = [str(i) for i in range(1000,1000+n)]
  D_bits = chars_to_padded_binary(D)
  keywords = [[x] for x in D]
  query_time_he, query_time_ms = performBloomSwizzlersBasic(keywords, D_bits, 0, n_attempts)
  query_time_he_list.append(query_time_he)
  query_time_ms_list.append(query_time_ms)
  size_list.append(db_size)
  print()
  print("DB Size:",db_size,"query_time_HE:", query_time_he, "query_time_ms:", query_time_ms)

df = pd.DataFrame(zip(size_list, query_time_he_list, query_time_ms_list), columns = ['db_size', 'QueryTime_HE', 'QueryTime_MS'])
df.to_csv('bloom_swizzlers_exp2_1.csv')