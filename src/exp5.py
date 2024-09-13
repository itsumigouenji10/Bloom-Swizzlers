
import pandas as pd
import statistics
from preprocessing import performPreProcessing, performPreProcessingBasic
import math

def compute_means_and_sds(attempt_basic_list, attempt_or_list, attempt_and_list):
    # Calculate means
    basic_mean = statistics.mean(attempt_basic_list)
    or_mean = statistics.mean(attempt_or_list)
    and_mean = statistics.mean(attempt_and_list)
    
    # Calculate standard deviations
    basic_sd = statistics.stdev(attempt_basic_list)
    or_sd = statistics.stdev(attempt_or_list)
    and_sd = statistics.stdev(attempt_and_list)
    
    # Return the results in a dictionary
    return {
        'basic_mean': basic_mean,
        'basic_sd': basic_sd,
        'or_mean': or_mean,
        'or_sd': or_sd,
        'and_mean': and_mean,
        'and_sd': and_sd
    }

def performComputationExperiment(keywords1, keywords, R=8):
  """
  This function creates an interface in Python Terminal, for performing AND and OR Bloom Swizzlers queries.
  """
  #print(keywords)

  basic_keywords = [[keyword] for keyword in keywords1]
  _, _, pre_processing_time_basic, basic_attempts = performPreProcessingBasic(basic_keywords)  
  _, _, pre_processing_time_and, pre_processing_time_or, and_attempts, or_attempts = performPreProcessing(keywords,R)
  return basic_attempts, and_attempts, or_attempts
    


db_size_range = [2**i for i in range(4,7)]
size_list = []
n_attempts = 2
average_attempt_basic_list = []
average_attempt_or_list = []
average_attempt_and_list = []

sd_attempt_basic_list = []
sd_attempt_or_list = []
sd_attempt_and_list = []

for db_size in db_size_range:
  n1 = int(math.sqrt(db_size))
  n2 = n1+1
  keywords1 = [str(i) for i in range(1000,1000+n1)]
  keywords2 = [str(i) for i in range(2000,2000+n2)]
  keywords = []
  for i in keywords1:
    for j in keywords2:
        keywords.append((i,j))

  size_list.append(db_size)
  attempt_basic_list = []
  attempt_or_list = []
  attempt_and_list = []
  for j in range(0,n_attempts):
      basic_attempts, and_attempts, or_attempts = performComputationExperiment(keywords1, keywords,R=16)
      attempt_basic_list.append(basic_attempts)
      attempt_and_list.append(and_attempts)
      attempt_or_list.append(or_attempts)

  
  means_and_sds = compute_means_and_sds(attempt_basic_list, attempt_or_list, attempt_and_list)
  average_attempt_basic_list.append(means_and_sds['basic_mean'])
  sd_attempt_basic_list.append(means_and_sds['basic_sd'])

  average_attempt_and_list.append(means_and_sds['and_mean'])
  sd_attempt_and_list.append(means_and_sds['and_sd'])

  
  average_attempt_or_list.append(means_and_sds['or_mean'])
  sd_attempt_or_list.append(means_and_sds['or_sd'])

  print()
  print("DB Size:",db_size,"Average Basic Attempts",means_and_sds['basic_mean'] , "Average AND Attempts",means_and_sds['and_mean'] , "Average OR Attempts",means_and_sds['or_mean'], "SD of Basic Attempts", means_and_sds['basic_sd'] , "SD of AND Attempts",means_and_sds['and_sd'] , "SD of OR Attempts",means_and_sds['or_sd'] ) 

df = pd.DataFrame(zip(size_list,  average_attempt_basic_list, average_attempt_and_list, average_attempt_or_list, sd_attempt_basic_list, sd_attempt_and_list, sd_attempt_or_list), columns = ['db_size', 'Mean - Basic', 'Mean - AND', 'Mean - OR', 'SD - Basic', 'SD - AND', 'SD - OR'])
df.to_csv("bloom_swizzlers_experiment5_results.csv")
