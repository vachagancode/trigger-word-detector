import torch

from utils import train
from dataset import TriggerWordDataset
from config import get_config

def start_training(annotations_file : str = "./annotations_file.csv", m=None, num_training=0):
    ds = TriggerWordDataset(annotations_file)
    cfg = get_config()

    # start the training
    model = train(cfg=cfg, dataset=ds, m=m)

    # save the last model
    torch.save(
        obj=model,
        f=f"modelv{num_training}.pth"
    )

    print(f"Model training finished")