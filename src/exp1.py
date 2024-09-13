from preprocessing import performPreProcessing, performPreProcessingBasic
import pandas as pd


def performExperiment1(keywords1, keywords, R=8):
  """
  This function creates an interface in Python Terminal, for performing AND and OR Bloom Swizzlers queries.
  """
  #print(keywords)

  basic_keywords = [[keyword] for keyword in keywords1]
  _, _, pre_processing_time_basic, basic_attempts = performPreProcessingBasic(basic_keywords)  
  _, _, pre_processing_time_and, pre_processing_time_or, and_attempts, or_attempts = performPreProcessing(keywords,R)
  return pre_processing_time_basic, pre_processing_time_and, pre_processing_time_or
    


db_size_range = [2**i for i in range(4,10)]
size_list = []
n_attempts = 3
pre_time_basic_list = []
pre_time_or_list = []
pre_time_and_list = []

for db_size in db_size_range:
  n1 = db_size
  keywords1 = [str(i) for i in range(1000,1000+n1)]
  keywords2 = [str(i) for i in range(2000,2000+n1)]
  keywords = []
  for i in range(0,len(keywords1)):
        keywords.append((keywords1[i],keywords2[i]))
  pre_time_or = 0
  pre_time_and = 0
  pre_time_basic = 0
  for j in range(0,n_attempts):
      pre_processing_time_basic, pre_processing_time_and, pre_processing_time_or = performExperiment1(keywords1, keywords,R=16)
      pre_time_basic += pre_processing_time_basic
      pre_time_or += pre_processing_time_or
      pre_time_and += pre_processing_time_and
  pre_time_basic = pre_time_basic/n_attempts   
  pre_time_or = pre_time_or/n_attempts
  pre_time_and = pre_time_and/n_attempts
  size_list.append(db_size)
  pre_time_basic_list.append(pre_time_basic)
  pre_time_or_list.append(pre_time_or)
  pre_time_and_list.append(pre_time_and)
  print()
  print("DB Size:",db_size,"PreProcessingTime_Basic", pre_time_basic, "PreProcessingTime_AND:",pre_time_and,"PreProcessingTime_OR:", pre_time_or) 

df = pd.DataFrame(zip(size_list,  pre_time_basic_list, pre_time_or_list, pre_time_and_list), columns = ['db_size', 'PreprocessingTime_Basic','PreprocessingTime_OR', 'PreprocessingTime_AND'])
df.to_csv('bloom_swizzlers_exp1_1.csv')
