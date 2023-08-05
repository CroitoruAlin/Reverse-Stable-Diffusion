import copy

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import Normalizer
from tqdm import tqdm

from transductive_kernel.extract_ensemble_preds import root_dataset


def get_kernel_matrix(set1, set2, gamma=0.01):
    kernel_matrix = np.zeros((len(set1), len(set2)))
    for i, elem1 in enumerate(tqdm(set1)):
        for j, elem2 in enumerate(set2):
            kernel_matrix[i][j] = np.dot(elem1, elem2)

    for i, elem1 in enumerate(set1):
        for j, elem2 in enumerate(set2):
            kernel_matrix[i][j] = kernel_matrix[i][j] / np.sqrt(kernel_matrix[i][i] * kernel_matrix[j][j])
        kernel_matrix = np.exp(-gamma*(1 - kernel_matrix) / 2)
    return kernel_matrix

#
kmeans = joblib.load("./kmeans.joblib")

clusters = kmeans.cluster_centers_
train_data = joblib.load("train_data.joblib")

closest, _ = pairwise_distances_argmin_min(clusters, train_data)
train_data = clusters
target_embeddings = joblib.load("sentence_embeddings.joblib")[closest]#

joblib.dump(target_embeddings, "target_embeddings.joblib")
target_embeddings = joblib.load("target_embeddings.joblib")
dict_test_data = joblib.load(f"{root_dataset}/ensemble_predictions.joblib")
test_data = []
target_test_data = []
images = []
for image in tqdm(dict_test_data):
    test_data.append(dict_test_data[image])
    target_test_data.append(np.load(f"{root_dataset}/sentence_embeddings/{image}.npy")
                            .squeeze())
    images.append(image)
#
test_data = np.array(test_data)
target_test_data = np.array(target_test_data)
normalizer = Normalizer()
train_data = normalizer.transform(train_data)
test_data = normalizer.transform(test_data)
data = np.concatenate((train_data, test_data), axis=0)
# print("Computing kernel matrix")
kernel_matrix = get_kernel_matrix(data, data)

joblib.dump(kernel_matrix, "kernel_matrix.joblib")
target_embeddings = joblib.load("target_embeddings.joblib")
train_matrix = kernel_matrix[:10000, :]
test_matrix = kernel_matrix[10000:, :]

print("Start training")

kernel_ridge = KernelRidge(alpha=10, kernel='linear', #gamma=0.01
                            )
kernel_ridge.fit(train_matrix, target_embeddings)
prediction = kernel_ridge.predict(test_matrix)

cosiine_similarities = []
for i, target in enumerate(target_test_data):
    cosiine_similarities.append(np.dot(target, prediction[i])/(np.linalg.norm(target) * np.linalg.norm(prediction[i]) +1e-10))

joblib.dump(kernel_ridge, "krr.joblib")
output = {}
for i, image in enumerate(images):
    output[image] = prediction[i]
joblib.dump(output,"features_krr.joblib")
print(f"Cosine similarity:{np.mean(cosiine_similarities)}")