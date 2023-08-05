from swin_image_experiments.train import train_swin_images
from swin_image_experiments.train_cl import train_swin_images_cl


class CFG:
    cl = False


if __name__ == "__main__":
    if CFG.cl:
        train_swin_images_cl()
    else:
        train_swin_images()