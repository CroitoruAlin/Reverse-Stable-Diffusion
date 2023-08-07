# Reverse Stable Diffusion: What prompt was used to generate this image?
### Introduction

This repository contains implements the training procedure introduced in: "Reverse Stable Diffusion: What prompt was
used to generate this image?" on the image-to-text-embedding task: https://arxiv.org/abs/2308.01472

### Prerequisites
Create a conda environment and run pip install:
```bash
conda create -n <name_env> python=3.9
conda activate <name_env>
pip install -r requirements.txt
```
The code expects a data set of image and text pairs, stored as follows:
```bash
|root_dir
  |images_part1
    |images
      |000000.png
  ...
  |images_part8
  |sentence_embeddings
    000000.npy
  metadata.csv
```
where ```sentence_embeddings``` is a directory and stores the target embeddings obtained from a sentence transformer.
In our experiments we used the following work to extract the embeddings: https://arxiv.org/pdf/1908.10084.pdf.


Moreover, it requires a vocabulary for multi-label classification. The script ```compute_vocab.py``` computes this vocabulary.

metadata.csv contains the pairs of images and text prompts.

The path to "root_dir" should be specified in global_configs.py.
### Train models

We have two scripts for each model to perform the training. The first one runs the vanilla training process, while 
the second one runs the curriculum learning procedure. These scripts can be invoked via main.py from each directory.

A special case is the U-Net because it expects the captions and it also works in the latent space of StableDiffusion, thus it requires
a preliminary step to map the images in this latent space(not included in the repo).
For captioning we used BLIP: https://github.com/salesforce/BLIP

### 


