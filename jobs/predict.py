import os
import sys
import csv
import logging

import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


logger = logging.getLogger()


@hydra.main(config_path="../config", config_name="predict")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    assert cfg.output_format in ["csv", "npz"], cfg.output_format

    logger.info("load emb paths")
    id2emb = {
        int(name.split(".")[0]): os.path.join(cfg.embeddings_dir, name)
        for name in tqdm.tqdm(os.listdir(cfg.embeddings_dir))
    }

    logger.info("load tracks")
    df = pd.read_csv(cfg.data_path).head(cfg.num_examples)
    n = len(df)
    logger.info(f"num tracks: {n}")

    ds = hydra.utils.instantiate(cfg.dataset)
    ds.data = [id2emb[int(track)] for track in df["track"]]

    collator = hydra.utils.instantiate(cfg.collator)

    logger.info("setup model")
    # load config from .hydra folder
    train_cfg = OmegaConf.load(os.path.join(os.path.dirname(cfg.checkpoint_path), "..", ".hydra", "config.yaml"))
    model = hydra.utils.instantiate(train_cfg.model)
    model.load_state_dict(torch.load(cfg.checkpoint_path, map_location="cpu"))
    model.to(cfg.device).eval()

    logger.info("predict")
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    y_pred = np.zeros((n, cfg.num_classes))
    i = 0
    with torch.no_grad(), tqdm.tqdm(total=n) as pbar:
        for batch in loader:
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            logits = model(**batch)  # [N, C]
            probs = torch.sigmoid(logits)
            j = min(i + cfg.batch_size, n)
            y_pred[i:j] = probs.to("cpu").numpy()
            pbar.update(j - i)
            i = j

    logger.info(f"save predictions in {cfg.output_format} format to {cfg.output_path}")
    if cfg.output_format == "csv":
        tracks = df["track"].tolist()
        with open(cfg.output_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["track", "prediction"])
            for i in tqdm.trange(n):
                writer.writerow([str(tracks[i]), ",".join(map(str, y_pred[i]))])
    elif cfg.output_format == "npz":
        np.savez(cfg.output_path, tracks=df["track"].astype(int).values, probs=y_pred)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
