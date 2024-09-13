import tenseal as ts
# For noting down Time
import time
import pandas as pd
from preprocessing import performPreProcessing
from querying import queryWithHE, decrypt_vector
from utils import chars_to_padded_binary_HE


def performBloomSwizzlers(keywords, D, D_bits, idx, R=8):
  """
  This function creates an interface in Python Terminal, for performing AND and OR Bloom Swizzlers queries.
  """
  flag, result, preprocessing_time_and, preprocessing_time_or, and_attempts, or_attempts = performPreProcessing(keywords, R)
  if flag is True:
    I_AND,I_AND_values, key_values = result[0], result[1],result[2]
    context = ts.context(
    ts.SCHEME_TYPE.BFV,
    poly_modulus_degree=8192,
    plain_modulus=1032193  # A prime number greater than your maximum value
)
    context.generate_galois_keys()
    context.generate_relin_keys()
    start_time = time.time()
    
    encrypted_res = queryWithHE(keywords[idx], I_AND, I_AND_values, key_values,len(keywords), keywords, context, D_bits)
    query_time_he_and = time.time() - start_time
    decrypted_result = [int(x)%2 for x in decrypt_vector(encrypted_res, context)]
    
    return preprocessing_time_and, query_time_he_and
    
n = 512
dependency_size_range = [0]
n_attempts = 1
query_time_he_and_list = []
preprocessing_time_and_list = []
size_list = []

for dependency_size in dependency_size_range:
  n1 = int(dependency_size*n)
  keywords1 = [str(1000) for i in range(0,n1)]
  i = 1001
  while len(keywords1) < n:
    keywords1.append(str(i))
    i = i+1
  i = i+1
  n2 = n
  keywords2 = [str(x) for x in range(i,i+n2)]
  db_size = n
  keywords = []
  for i in range(0,len(keywords1)):
    keywords.append((keywords1[i],keywords2[i]))
  D = [str('c')+str(i) for i in range(0,len(keywords))]
  D_bits = chars_to_padded_binary_HE(D)
  preprocessing_time_and, query_time_he_and = performBloomSwizzlers(keywords, D, D_bits, 0, R=8)
  preprocessing_time_and_list.append(preprocessing_time_and)
  query_time_he_and_list.append(query_time_he_and)
  size_list.append(db_size)
  print("Dependency:", dependency_size*100, "% n1: ",n1, "n2:",n2,"DB Size:", len(keywords), "Processing Time (AND):", preprocessing_time_and, " Querying Time with HE (AND):", query_time_he_and )

df = pd.DataFrame(zip(dependency_size_range, size_list, preprocessing_time_and_list, query_time_he_and_list), columns = ['dependency_size', 'db_size', 'PreProcessingTime_AND', 'QueryTime_AND'])
df.to_csv('bloom_swizzlers_exp1_2.csv')