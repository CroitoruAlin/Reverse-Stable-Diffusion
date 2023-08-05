import os

import joblib
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm

from global_configs import root_dir, csv_name

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
df = pd.read_csv(os.path.join(root_dir, csv_name))
frequency = []
word_id = {}
id_word = {}
id = 0
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend([".", ",", "!"])
for ind in tqdm(df.index):
    prompt = df['prompt'][ind]
    words = nltk.tag.pos_tag(nltk.word_tokenize(prompt))

    for word, pos in words:
        if "JJ" not in pos and \
            "NN" not in pos and \
            "VB" not in pos:
            continue
        word = word.lower()
        if word.lower() not in stopwords and word.isalpha():
            if word in word_id:
                frequency[word_id[word]]+=1
            else:
                frequency.append(1)
                word_id[word]=id
                id_word[id]=word
                id+=1

sorted_indices = np.argsort(frequency)
vocab = {}
id = 0
for index in sorted_indices[-1000:]:
    word = id_word[index]
    vocab[word] = id
    id+=1
joblib.dump(vocab,os.path.join(root_dir, 'vocab.joblib'))

