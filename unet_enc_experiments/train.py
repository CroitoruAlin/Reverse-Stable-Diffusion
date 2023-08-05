import json
from collections.abc import Iterable

import joblib
import numpy as np
import skimage
import timm
import torch
from einops import rearrange
from scipy import spatial
from timm.utils import AverageMeter
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import VisionEncoderDecoderModel, AutoTokenizer

from data_handler import get_dataloaders
from stablediffusion.ldm.modules.diffusionmodules.encoder_unet import EncoderUNetModel
from sentence_transformers import SentenceTransformer
from stablediffusion.ldm.modules.diffusionmodules.openaimodel import UNetModel
from stablediffusion.ldm.modules.encoders.modules import FrozenOpenCLIPEmbedder
from swin_image_experiments.loss import BCECossLoss
from unet_enc_experiments.losses import HuberCosLoss, ClipLoss
from unet_enc_experiments.model import UnetModel


def cosine_similarity(y_trues, y_preds):
    return np.mean([
        1 - spatial.distance.cosine(y_true, y_pred)
        for y_true, y_pred in zip(y_trues, y_preds)
    ])

def cosine_similarity_individual(y_trues, y_preds):
    return [1 - spatial.distance.cosine(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)]

def train(
        trn_df,
        val_df,
        CFG
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    dataloaders = get_dataloaders(
        trn_df,
        val_df,
        CFG
    )

    conf = OmegaConf.load("../stablediffusion/scripts/configs/stable-diffusion/v2-train.yaml")
    model = UnetModel(conf["model"].params.unet_config.params)
    state_dict = torch.load("unet_model.pth")
    model.unet.load_state_dict(state_dict, strict=False)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)

    ttl_iters = CFG.num_epochs * len(dataloaders['train'])
    scheduler = CosineAnnealingLR(optimizer, T_max=ttl_iters, eta_min=1e-6)

    criterion = BCECossLoss(w=0.1)
    best_score = -1.0
    st_model = SentenceTransformer(
        '../all-MiniLM-L6-v2',
        device=device)

    st_model.eval()

    transformer_stable_diffusion = FrozenOpenCLIPEmbedder(freeze=True, layer='penultimate')
    transformer_stable_diffusion.load_state_dict(torch.load("transformer_stable_diffusion.pth"))
    transformer_stable_diffusion.eval()
    transformer_stable_diffusion.to(device)

    cos_errors = {}
    # uc = np.load("./unconditional.npy")
    dataloaders['train'].dataset.use_entire_dataset()
    dataloaders['val'].dataset.use_entire_dataset()
    for epoch in range(CFG.num_epochs):
        train_meters = {
            'loss': AverageMeter(),
            'cos': AverageMeter(),
        }
        model.train()
        pbar = tqdm(dataloaders['train'], leave=False)
        for X, y, timesteps, caption, classes, _ in pbar:

            caption = transformer_stable_diffusion.encode(caption)
            X, y = X.to(device).float(), y.to(device).float()
            classes = classes.to(device)
            caption = torch.Tensor(caption).to(device).float()
            optimizer.zero_grad()
            timesteps = timesteps.to(device).float()
            X_out, pred_classes = model(X, timesteps=timesteps, context=caption)
            target = torch.ones(X_out.size(0)).to(device)
            loss = criterion(X_out, y, target, preds = pred_classes, classes=classes)
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
        #
        val_meters = {
            'loss': AverageMeter(),
            'cos': AverageMeter(),
        }
        model.eval()
        for X, y, timesteps, caption, classes, _ in tqdm(dataloaders['val'], leave=False):
            caption = transformer_stable_diffusion.encode(caption)
            X, y = X.to(device).float(), y.to(device).float()
            classes = classes.to(device)
            caption = torch.Tensor(caption).to(device).float()
            with torch.no_grad():
                timesteps = timesteps.to(device).float()
                X_out, pred_classes = model(X, timesteps=timesteps, context=caption)
                target = torch.ones(X_out.size(0)).to(device)
                loss = criterion(X_out, y, target, preds = pred_classes, classes=classes)

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
            torch.save(model.state_dict(), f'../trained_models/{CFG.model_name}.pth')

            with torch.no_grad():
                for X, y, timesteps, caption, classes, img_path in tqdm(dataloaders['train'], leave=False):
                    caption = transformer_stable_diffusion.encode(caption)
                    X, y = X.to(device).float(), y.to(device).float()
                    caption = torch.Tensor(caption).to(device).float()
                    optimizer.zero_grad()
                    timesteps = timesteps.to(device).float()

                    X_out, _ = model(X, timesteps=timesteps, context=caption)

                    val_loss_cos = cosine_similarity_individual(X_out.detach().cpu().numpy(), y.detach().cpu().numpy())

                    for i, ipath in enumerate(img_path):
                        if ipath in cos_errors:
                            cos_errors[ipath].append(val_loss_cos[i].item())
                        else:
                            cos_errors[ipath] = [val_loss_cos[i].item()]
                joblib.dump(cos_errors, "cos_errors.joblib")
