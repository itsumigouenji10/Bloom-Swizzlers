import pandas as pd
from preprocessing import performPreProcessingAll
from querying import queryUsingBloomSwizzlers
from utils import convert_sentences_to_binary

def main():
    # Initialize data
    keywords1 = ['w', 'x', 'y', 'z']
    keywords2 = ['a', 'b', 'c', 'd']
    keywords = list(zip(keywords1, keywords2))
    basic_keywords = [[keyword[1]] for keyword in keywords]
    D = ['v'+str(i+1) for i in range(len(keywords))]
    D_bits = convert_sentences_to_binary(D)
    GLOBAL_KEY = b'10000000\x06O\x9e\xf0\xa2\xfd\x834'

    # Use this to trigger the actual functionality
    queryUsingBloomSwizzlers(keywords, basic_keywords, D, D_bits, GLOBAL_KEY)

if __name__ == "__main__":
    main()
