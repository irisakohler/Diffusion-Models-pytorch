FROM nvcr.io/nvidia/pytorch:23.02-py3

RUN pip3 install kaggle \
    wandb \
    Pillow \
    numpy \
    matplotlib \
    fastai \
    fastdownload \
    fastprogress \
    fastcore \
    tqdm