import tenseal as ts
import time
import pandas as pd
import math
from preprocessing import performPreProcessing
from querying import queryWithHE
from utils import chars_to_padded_binary_HE

def performBloomSwizzlers(keywords, D, D_bits, idx, n_attempts, R=8):
  """
  This function creates an interface in Python Terminal, for performing AND and OR Bloom Swizzlers queries.
  """
  flag, result, preprocessing_time_and, preprocessing_time_or, and_attempts, or_attempts = performPreProcessing(keywords, R)
  if flag is True:
    I_OR, I_AND,I_OR_values, I_AND_values, key_values, or_keywords = result[0], result[1],result[2], result[3], result[4], result[5]
    context = ts.context(
    ts.SCHEME_TYPE.BFV,
    poly_modulus_degree=8192,
    plain_modulus=1032193  
)
    context.generate_galois_keys()
    context.generate_relin_keys()
    query_time_he_and = 0
    query_time_he_or = 0
    for j in range(0,n_attempts):
        start_time = time.time()
        encrypted_res = queryWithHE(keywords[idx], I_AND, I_AND_values, key_values,len(keywords), keywords, context, D_bits)
        query_time_he_and += time.time() - start_time
        start_time = time.time()
        encrypted_res = queryWithHE(keywords[idx], I_OR, I_OR_values, key_values,len(or_keywords), or_keywords, context, D_bits)
        query_time_he_or += time.time() - start_time
    return query_time_he_and/n_attempts, query_time_he_or/n_attempts, preprocessing_time_and, preprocessing_time_or
    


db_size_range = [2**i for i in range(4,14)]
size_list = []
n_attempts = 3
query_time_he_and_list = []
query_time_he_or_list = []

for db_size in db_size_range:
  n1 = int(math.sqrt(db_size))
  n2 = n1+1
  keywords1 = [str(i) for i in range(1000,1000+n1)]
  keywords2 = [str(i) for i in range(2000,2000+n2)]
  keywords = []
  for i in keywords1:
    for j in keywords2:
        keywords.append((i,j))
  D = [str('c')+str(i) for i in range(0,len(keywords))]
  D_bits = chars_to_padded_binary_HE(D)
  query_time_he_and, query_time_he_or, preprocessing_time_and, preprocessing_time_or = performBloomSwizzlers(keywords, D, D_bits, 0, n_attempts, R=8)
  size_list.append(db_size)
  query_time_he_and_list.append(query_time_he_and)
  query_time_he_or_list.append(query_time_he_or)
  print()
  print("DB Size:",db_size,"QueryTime_HE_AND", query_time_he_and, "QueryTime_HE_OR:",query_time_he_or) 

df = pd.DataFrame(zip(size_list,  query_time_he_and_list, query_time_he_or_list), columns = ['db_size', 'QueryTime_HE_AND','QueryTime_HE_OR'])
df.to_csv('bloom_swizzlers_exp2_2.csv')









