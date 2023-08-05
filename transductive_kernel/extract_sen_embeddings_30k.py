import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from transductive_kernel.extract_ensemble_preds import root_dataset

st_model = SentenceTransformer(
        '../all-MiniLM-L6-v2',
        device="cuda")

st_model.eval()
dtype = {'prompt': str}
prompts = pd.read_csv(os.path.join(root_dataset, "prompts.csv"), dtype=dtype)


# Iterate through rows
for row in tqdm(prompts.values):

    file_name = row[1].split("/")[-1].split(".")[0]
    prompt = row[2]
    if not isinstance(prompt, str):
        prompt = "nan"
    try:
        encoding = st_model.encode([prompt])
    except:
        print(1)
        continue
    np.save(f"{root_dataset}/sentence_embeddings/{file_name}.npy", encoding)

