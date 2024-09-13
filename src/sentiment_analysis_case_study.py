import tenseal as ts
# For noting down Time
import time
import pandas as pd
from preprocessing import performPreProcessing
from querying import  decrypt_vector, encrypt_vector, homomorphic_matrix_multiplication
import numpy as np
import siphash

def queryWithHE(queries, I, I_values, key_values,m, keywords, context):
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
  return q

def performBloomSwizzlers(keywords, idx, username, R=8):
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
    start_time = time.time()
    
    input_keywords = (username, 'Positive')
    encrypted_res = queryWithHE(input_keywords, I_AND, I_AND_values, key_values,len(or_keywords), or_keywords, context)
    query_time_he_and = time.time() - start_time
    decrypted_result = [int(x)%2 for x in decrypt_vector(encrypted_res, context)]
    positive_count = decrypted_result.count(1)

    print('SELECT COUNT(*) WHERE Username = ', input_keywords[0], 'AND Polarity =', input_keywords[1])
    print('Result:', positive_count)


    input_keywords = (username, 'Negative')
    encrypted_res = queryWithHE(input_keywords, I_AND, I_AND_values, key_values,len(or_keywords), or_keywords, context)
    query_time_he_and = time.time() - start_time
    decrypted_result = [int(x)%2 for x in decrypt_vector(encrypted_res, context)]
    negative_count = decrypted_result.count(1)

    total_count = positive_count + negative_count
    print('SELECT COUNT(*) WHERE Username = ', input_keywords[0], 'AND Polarity =', input_keywords[1])
    print('Negative Tweets Count:', negative_count)
    print()
    print('Sentiment Analysis for Username = ', input_keywords[0])
    print('Positive Tweets: ', int(positive_count*100/total_count), '%')
    print('Negative Tweets: ', int(negative_count*100/total_count), '%')

def countScores(usernames, scores):
    scores_list = []
    username_unique = []
    scores_count = []
    count_dict = {}
    for username in usernames:
      count_dict[username] = {}
      count_dict[username][0] = 0
      count_dict[username][4] = 1
    for i in range(0,len(usernames)):
      if scores[i] == 0:
        count_dict[usernames[i]][0] += 1
      else:
        count_dict[usernames[i]][4] += 1
    for username in count_dict:
      scores_list.append('Negative')
      username_unique.append(username)
      scores_count.append(count_dict[username][0])
      scores_list.append('Positive')
      username_unique.append(username)
      scores_count.append(count_dict[username][4])
    combined = list(zip(scores_list, username_unique, scores_count))
    sorted_combined = sorted(combined, key=lambda x: x[2], reverse=True)
    scores_list = []
    username_unique = []
    scores_count = []
    for i in range(0,len(sorted_combined)):
      scores_list.append(sorted_combined[i][0])
      username_unique.append(sorted_combined[i][1])
      scores_count.append(sorted_combined[i][2]) 
    return scores_list, username_unique, scores_count


n_attempts = 3
n = 10
idx = 0
tweets_df = pd.read_csv('./tweets.csv', encoding='ISO-8859-1', names = ['score', 'tweet_id', 'date', 'query', 'username', 'tweet'])
tweets_df = tweets_df.sample(frac=1).reset_index(drop=True)


usernames = tweets_df['username'].tolist()
scores = tweets_df['score'].tolist()
polarities = []
for score in scores:
  if score == 0:
    polarities.append('Negative')
  else:
    polarities.append('Positive')

def generate_list(usernames, polarities):
    combined = []
    for i in range(0,len(usernames)):
      combined.append((usernames[i], polarities[i]))
    combined.sort(key=lambda x: x[0])
    result = []
    count_dict = {}
    for username, score in combined:
        if len(result) >= 10:
            break
        if username not in count_dict:
            count_dict[username] = 0
        if (username == list(count_dict.keys())[0] and count_dict[username] < 3) or (username != list(count_dict.keys())[0] and count_dict[username] < 2):
            result.append([username, score])
            count_dict[username] += 1
    result[2][1] = 'Negative'
    return result
result_list = generate_list(usernames, polarities)

keywords = []
for i in range(0, n):
  keywords.append((str(result_list[i][0]), str(result_list[i][1])))


print("\nDatabase:")
columns = []
for i in range(len(keywords[0])):
    columns.append(f"Attribute {i + 1}")
db_df = pd.DataFrame(keywords, columns=columns)
print(db_df.to_string(index=False))
print()

print("Enter username: ", end = "")
username = input()
print()

performBloomSwizzlers(keywords, idx, username, R=8)
