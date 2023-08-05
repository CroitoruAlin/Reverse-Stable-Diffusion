from clip_experiments.train import train_clip
from clip_experiments.train_cl import train_clip_cl


class CFG:
    cl = True

if __name__ == "__main__":
    if CFG.cl:
        train_clip_cl()
    else:
        train_clip()

