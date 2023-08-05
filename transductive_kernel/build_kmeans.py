import os.path

import joblib
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans
from tqdm import tqdm

from global_configs import root_dir

ensemble_preds = joblib.load("%sensemble_predictions.joblib" % root_dir)

train_data = []
sentence_embeddings = []
for image in tqdm(ensemble_preds):
    train_data.append(ensemble_preds[image])
    sentence_embeddings.append(np.load(os.path.join(root_dir, "sentence_embeddings", f"{image}npy.npy")))

train_data = np.array(train_data)
sentence_embeddings = np.array(sentence_embeddings)
kmeans = BisectingKMeans(n_clusters=10000, init="k-means++", verbose=1)
kmeans.fit(train_data)
joblib.dump(kmeans, "kmeans.joblib")
joblib.dump(sentence_embeddings, "sentence_embeddings.joblib")
joblib.dump(train_data, "train_data.joblib")
