import siphash
import tenseal as ts
import pandas as pd
import numpy as np
from preprocessing import performPreProcessing
from querying import encrypt_vector, homomorphic_matrix_multiplication, decrypt_vector
from utils import single_padded_binary_to_char, chars_to_padded_binary

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
      e[i] = e[i] + e_values[j][i] - 2*e[i]*e_values[j][i]
  I = I.astype(float)
  q = homomorphic_matrix_multiplication(e, I, context)
  encrypted_res = homomorphic_matrix_multiplication(q, D_bits, context)  
  return encrypted_res

def performBloomSwizzlers(keywords, D_dep_time, D_bits_dep_time, D_arr_time, D_bits_arr_time, keyword, R=8):
  """
  This function creates an interface in Python Terminal, for performing AND and OR Bloom Swizzlers queries.
  """
  flag, result, preprocessing_time_and, preprocessing_time_or, and_attempts, or_attempts = performPreProcessing(keywords, R)
  if flag is True:
    I_OR, I_AND,I_OR_values, I_AND_values, key_values, or_keywords = result[0], result[1],result[2], result[3], result[4], result[5]
    context = ts.context(
    ts.SCHEME_TYPE.BFV,
    poly_modulus_degree=8192,
    plain_modulus=1032193  # A prime number greater than your maximum value
)
    context.generate_galois_keys()
    context.generate_relin_keys()
    
    print()
    print()
    print('Querying for Departure Time:')
    print()
    encrypted_res = queryWithHE(keyword, I_AND, I_AND_values, key_values,len(or_keywords), or_keywords, context, D_bits_dep_time)
    decrypted_result = [int(x)%2 for x in decrypt_vector(encrypted_res, context)]
    res_idx = single_padded_binary_to_char(decrypted_result, D_bits_dep_time, D_dep_time)
    print('SELECT departure_time WHERE origin = ', keyword[0], ' AND destination = ', keyword[1])
    departure_time = D_dep_time[res_idx]
    print('Received departure_time = ', departure_time)

    print()
    print()
    print('Querying for Arrival Time:')
    print()
    encrypted_res = queryWithHE(keyword, I_AND, I_AND_values, key_values,len(or_keywords), or_keywords, context,  D_bits_arr_time)
    decrypted_result = [int(x)%2 for x in decrypt_vector(encrypted_res, context)]
    res_idx = single_padded_binary_to_char(decrypted_result, D_bits_arr_time, D_arr_time)
    print('SELECT arrival_time WHERE origin = ', keyword[0], ' AND destination = ', keyword[1])
    arrival_time = D_arr_time[res_idx]
    print('Received arrival_time = ', arrival_time)

    print()
    print('Flight duration: ', calculate_flight_duration(departure_time, arrival_time))
    print()
    print()
    return preprocessing_time_and, preprocessing_time_or
    
import random
random.seed(0)

from datetime import datetime

def calculate_flight_duration(departure_time, arrival_time):
    time_format = "%H:%M"
    departure_dt = datetime.strptime(departure_time, time_format)
    arrival_dt = datetime.strptime(arrival_time, time_format)
    duration = arrival_dt - departure_dt
    if duration.days < 0:
        duration = (datetime.strptime("24:00", time_format) - departure_dt) + (arrival_dt - datetime.strptime("00:00", time_format))
    total_seconds = duration.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    return f"{hours:02} hours and {minutes:02} minutes"



n_attempts = 3
n = 10
idx = 0
flight_df = pd.read_csv('./flight_schedule.csv', encoding='ISO-8859-1', header=0, )
flight_df = flight_df.sample(frac=1).reset_index(drop=True)
flight_df = flight_df.dropna()
origins = list(flight_df['origin'])
destinations = list(flight_df['destination'])
dep_time = list(flight_df['scheduledDepartureTime'])
arr_time = list(flight_df['scheduledArrivalTime'])


D_dep_time = [x for x in dep_time]
D_bits_dep_time = chars_to_padded_binary(D_dep_time[:n])

D_arr_time = [x for x in arr_time]
D_bits_arr_time = chars_to_padded_binary(D_arr_time[:n])


keywords = []
for i in range(0, n):
  keywords.append((str(origins[i]), destinations[i]))

print("\nDatabase:")
columns = []
columns.append('origin')
columns.append('destination')
columns.append('departure_time')
columns.append('arrival_time')
keyword_columns = list(zip(*keywords))
data = list(zip(*keyword_columns, D_dep_time, D_arr_time))
db_df = pd.DataFrame(data[:5], columns=columns)
print(db_df.to_string(index=False))


print()
print()
print("Enter origin: ", end = "")
origin = input()
print("Enter destination: ", end = "")
destination = input()
keyword = [origin, destination]

preprocessing_time_and, preprocessing_time_or = performBloomSwizzlers(keywords, D_dep_time, D_bits_dep_time, D_arr_time, D_bits_arr_time, keyword, R=8)
# # print("DB Size:", len(keywords), "Processing Time (AND):", preprocessing_time_and, " Querying Time with HE (AND):", query_time_he_and, "Processing Time (OR):", preprocessing_time_or, " Querying Time with HE (OR):", query_time_he_or, )
