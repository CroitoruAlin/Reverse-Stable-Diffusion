from vit_image_experiments.train import train_vit_images
from vit_image_experiments.train_cl import train_vit_images_cl


class CFG:
    cl=False


if __name__ =="__main__":
    if CFG.cl:
        train_vit_images_cl()
    else:
        train_vit_images()