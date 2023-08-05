import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import warnings

from global_configs import root_dir, csv_name
from unet_enc_experiments.train import train
from unet_enc_experiments.train_cl import train_cl

warnings.filterwarnings('ignore')

class CFG:
    model_name = "unet_enc_multilabel"
    batch_size = 64
    num_epochs = 10
    lr = 1e-4
    seed = 42
    use_noisy_examples = False
    cl = True

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
if __name__ == "__main__":

    root_dataset = os.path.join(root_dir,"/latent_features/images")
    df = pd.read_csv(os.path.join(root_dir, csv_name))
    captions_folder = os.path.join(root_dir, "captions")

    dict = df.to_dict()
    new_df = {'filepath': [], 'prompt': [], 'caption': []}
    for path in dict['filepath']:
        path_str = dict['filepath'][path]
        prompt = dict['prompt'][path]
        new_path = path_str.split("/")[-2:]
        dir = new_path[0]
        file_name = new_path[1].split(".")[0]
        new_path = os.path.join(root_dataset, file_name+".npy")
        new_df['filepath'].append(new_path)
        new_df['prompt'].append(dict['prompt'][path])
        new_df['caption'].append(os.path.join(captions_folder, file_name+".txt"))


    new_df = pd.DataFrame.from_dict(new_df)
    trn_df, val_df = train_test_split(new_df, test_size=0.1, random_state=CFG.seed)
    if CFG.cl:
        train_cl(trn_df, val_df, CFG)
    else:
        train(trn_df, val_df, CFG)
