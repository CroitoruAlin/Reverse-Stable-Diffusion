import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import spatial
from timm.utils import AverageMeter
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from global_configs import csv_name, root_dir
from swin_image_experiments.data_handler import get_dataloaders
from swin_image_experiments.loss import BCECossLoss
from swin_image_experiments.model import SwinModel
from utils.utils import cosine_similarity


class CFG:
    model_name = "swin_images_multi_label"
    input_size = 256
    batch_size = 64
    num_epochs = 3
    lr = 1e-4
    seed = 42

def cosine_similarity_individual(y_trues, y_preds):
    return [1 - spatial.distance.cosine(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)]


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def train_swin_images():
    df1 = pd.read_csv(os.path.join(root_dir, csv_name))
    train_dataloader, val_dataloader = get_dataloaders(df1, CFG)
    train_dataloader.dataset.use_entire_dataset()
    val_dataloader.dataset.use_entire_dataset()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SwinModel(use_multi_label_output=False)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)

    ttl_iters = CFG.num_epochs * len(train_dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=ttl_iters, eta_min=1e-5)
    criterion = BCECossLoss(w=0.1)

    best_score = -1.0
    cos_errors={}
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
            X_out, preds = model(X)
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
                X_out, preds = model(X)

                target = torch.ones(X.size(0)).to(device)
                loss = criterion(X_out, y, target,  preds, classes)

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
            with torch.no_grad():
                for X, y, classes, img_path in tqdm(train_dataloader, leave=False):

                    X, y = X.to(device).float(), y.to(device).float()
                    optimizer.zero_grad()

                    X_out, _ = model(X)

                    val_loss_cos = cosine_similarity_individual(X_out.detach().cpu().numpy(), y.detach().cpu().numpy())

                    for i, ipath in enumerate(img_path):
                        if ipath in cos_errors:
                            cos_errors[ipath].append(val_loss_cos[i].item())
                        else:
                            cos_errors[ipath] = [val_loss_cos[i].item()]
                joblib.dump(cos_errors, "cos_errors.joblib")


