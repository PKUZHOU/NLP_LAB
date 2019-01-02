import bcolz
import numpy as np
def load_embeddings(folder_path):
    folder_path = folder_path.rstrip('/')
    words = bcolz.carray(rootdir='%s/words' % folder_path, mode='r')
    embeddings = bcolz.carray(rootdir='%s/embeddings' % folder_path, mode='r')
    return words, embeddings

folder_path = 'zh_char.64'
words,embeddings = load_embeddings(folder_path)
words = list(words)
for i in range(100):
    print(words[i])
print(words,embeddings)