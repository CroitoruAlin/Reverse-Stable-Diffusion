import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import spatial
from sklearn.model_selection import train_test_split
from timm.utils import AverageMeter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW

from clip_experiments.data_handler import ClipEmbds
from clip_experiments.model import ClipMLP
from global_configs import root_dir, csv_name

from swin_image_experiments.loss import BCECossLoss


def cosine_similarity(y_trues, y_preds):
    return np.mean([
        1 - spatial.distance.cosine(y_true, y_pred)
        for y_true, y_pred in zip(y_trues, y_preds)
    ])
class CFG:
    model_name = "clip_huge_mlp_multilabel_cl"
    input_size = 224
    batch_size = 64
    num_epochs = 15
    lr = 1e-4
    seed = 42
    device = torch.device('cuda')


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def train_clip_cl():
    seed_everything(CFG.seed)
    root_dataset = root_dir
    df = pd.read_csv(os.path.join(root_dataset, csv_name), usecols=['filepath','prompt'])

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=CFG.seed)
    cos_errors = joblib.load("cos_errors.joblib")
    weighted_cos_error = {k: np.average(v, weights=[0., 0., 0., 0.2, 0.3, 0.5]) for k, v in cos_errors.items()}

    temp_df = pd.DataFrame.from_dict(
        {'paths': list(weighted_cos_error.keys()), 'cos': list(weighted_cos_error.values())})
    new_train_df = pd.merge(left=train_df, right=temp_df, how='inner', left_on='filepath',
                            right_on='paths')
    new_train_df.sort_values(by='cos', ascending=False, inplace=True)
    new_train_df.drop(columns=['paths', 'cos'], inplace=True)
    train_df = new_train_df


    train_dataset = ClipEmbds(root_dataset, train_df)

    val_dataset = ClipEmbds(root_dataset, val_df)
    val_dataset.use_entire_dataset()

    train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=CFG.batch_size, pin_memory=True, num_workers=4, drop_last=False)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=CFG.batch_size, pin_memory=True, num_workers=4, drop_last=False)

    model = ClipMLP()
    model.to(device=CFG.device)

    optimizer = AdamW(model.parameters(), lr=CFG.lr)

    ttl_iters = CFG.num_epochs * len(train_dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=ttl_iters, eta_min=1e-6)

    criterion = BCECossLoss(w=0.1)
    device = "cuda"
    best_score = 0.


    for epoch in range(CFG.num_epochs):
        train_meters = {
            'loss': AverageMeter(),
            'cos': AverageMeter(),
        }
        model.train()
        pbar = tqdm(train_dataloader, leave=False)
        for X, y, classes, _ in pbar:

            X, y, classes = X.to(device).float(), y.to(device).float(), classes.to(device).float()

            optimizer.zero_grad()
            X_out, preds = model.use_mlp(X)

            target = torch.ones(X.size(0)).to(device)
            loss = criterion(X_out, y, target, preds, classes)

            loss.backward()

            optimizer.step()
            scheduler.step()

            trn_loss = loss.item()
            trn_cos = cosine_similarity(
                X_out.detach().cpu().numpy(),
                y.detach().cpu().numpy()
            )

            train_meters['loss'].update(trn_loss, n=X.size(0))
            train_meters['cos'].update(trn_cos, n=X.size(0))

            pbar.set_description('Epoch {:d} / trn/loss={:.4f}, trn/cos={:.4f}'.format(
                epoch + 1,
                train_meters['loss'].avg,
                train_meters['cos'].avg))

        print('Epoch {:d} / trn/loss={:.4f}, trn/cos={:.4f}'.format(
            epoch + 1,
            train_meters['loss'].avg,
            train_meters['cos'].avg))

        val_meters = {
            'loss': AverageMeter(),
            'cos': AverageMeter(),
        }
        model.eval()
        for X, y, classes, _ in tqdm(val_dataloader, leave=False):

            X, y, classes = X.to(device).float(), y.to(device).float(), classes.to(device).float()

            with torch.no_grad():
                X_out, preds = model.use_mlp(X)

                target = torch.ones(X.size(0)).to(device)
                loss = criterion(X_out, y, target, preds, classes)

                val_loss = loss.item()
                val_cos = cosine_similarity(
                    X_out.detach().cpu().numpy(),
                    y.detach().cpu().numpy()
                )

            val_meters['loss'].update(val_loss, n=X.size(0))
            val_meters['cos'].update(val_cos, n=X.size(0))

        print('Epoch {:d} / val/loss={:.4f}, val/cos={:.4f}'.format(
            epoch + 1,
            val_meters['loss'].avg,
            val_meters['cos'].avg))

        if val_meters['cos'].avg > best_score:
            best_score = val_meters['cos'].avg
            torch.save(model.state_dict(),
                       f'../trained_models/{CFG.model_name}.pth')

        train_dataloader.dataset.update_stage()


