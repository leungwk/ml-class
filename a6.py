import numpy as np
import matplotlib.pyplot as plt
import re
import nltk.stem.porter
from nltk.tokenize import wordpunct_tokenize, sent_tokenize

# ('emailSample1.txt')
# ex6/emailSample1.txt

vocab = np.genfromtxt('data/ex6/vocab.txt', delimiter='\t', dtype=[("idx", int), ("word", object)], converters={1: str})
vocab = np.array((vocab['idx'], map(str, vocab['word'])))

# file_contents = readFile('emailSample1.txt');

# ~/src/ml-class/data/ex6/emailSample1.txt
def process_email():
    with open('data/ex6/emailSample1.txt') as f:
        lines = f.readlines()
        lines = [line.lower() for line in lines]
        lines = [re.sub('<[^<>]+>', ' ', line) for line in lines]
        lines = [re.sub('[0-9]+', 'number', line) for line in lines]
        lines = [re.sub('(http|https)://[^\s]*', 'httpaddr', line) for line in lines]
        lines = [re.sub('[^\s]+@[^\s]+', 'emailaddr', line) for line in lines]
        lines = [re.sub('[$]+', 'dollar', line) for line in lines]

        lines = [re.sub('[@$/#\.-:&\*\+=\[\]\?!\(\)\{\},\'">_<;%\n]', '', line) for line in lines]
        line = " ".join(lines)
        words = wordpunct_tokenize(line)
        
        ps = nltk.stem.porter.PorterStemmer()
        words = [ps.stem(w) for w in words]
    return words
        
res = process_email()
