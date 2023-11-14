import os
import sys
import logging

import torch
import torch.distributed as dist

import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig
import pandas as pd

import transformers
from transformers import EarlyStoppingCallback
from transformers.integrations import MLflowCallback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def get_logger(training_args):
    logger = logging.getLogger("training")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.local_rank in [-1, 0]:
        log_level = logging.INFO
    else:
        log_level = logging.WARN
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    return logger


# TODO: пока всё пишу для world_size == 1

@hydra.main(config_path="../config", config_name="train")
def main(cfg: DictConfig):
    args = hydra.utils.instantiate(cfg.training_args)
    logger = get_logger(args)
    logger.info(f"world size: {args.world_size}")
    logger.info(f"output_dir: {args.output_dir}")
    logger.info(f"hydra outputs dir: {os.getcwd()}")
    if args.local_rank in [-1, 0]:
        print(OmegaConf.to_yaml(cfg))

    logger.info("loading embeddings")
    # id2emb = {
    #     int(name.split(".")[0]): np.load(os.path.join(cfg.embeddings_dir, name))
    #     for name in tqdm.tqdm(os.listdir(cfg.embeddings_dir))
    # }
    id2emb = {
        int(name.split(".")[0]): os.path.join(cfg.embeddings_dir, name)
        for name in tqdm.tqdm(os.listdir(cfg.embeddings_dir))
    }

    logger.info("loading labels")
    df_train = pd.read_csv(cfg.train_data_path).head(cfg.num_train_examples)

    def reformat_data(df):
        data = []
        for i, row in enumerate(df.itertuples()):
            if (args.local_rank == -1) or (i % args.world_size == args.local_rank):
                data.append((int(row.track), id2emb[int(row.track)], [int(tag) for tag in row.tags.split(",")]))
        return data

    train_data = reformat_data(df_train)

    if args.world_size > 1:
        logger.info("syncing number of training examples...")
        n = torch.tensor(len(train_data)).to(args.device)
        ns = [torch.empty_like(n) for _ in range(args.world_size)]
        dist.all_gather(ns, n)
        logger.info(f"num training examples per process: {[x.item() for x in ns]}")
        dist.all_reduce(n, op=dist.ReduceOp.MIN)
        n = n.item()
        logger.info(f"synced number of training examples: {n}")
        train_data = train_data[:n]

    ds_train = hydra.utils.instantiate(cfg.training_dataset)
    ds_train.set_data(train_data)

    if cfg.training_args.do_eval:
        logger.info("load valid data")
        df_valid = pd.read_csv(cfg.valid_data_path).head(cfg.num_valid_examples)
        logger.info(f"num valid examples: {len(df_valid)}")
        valid_data = reformat_data(df_valid)
        ds_valid = hydra.utils.instantiate(cfg.valid_dataset)
        ds_valid.set_data(valid_data)
    else:
        ds_valid = None

    collator = hydra.utils.instantiate(cfg.collator)

    if cfg.checkpoint_path is not None:
        # 1. подгрузить конфиг, из которого была создана модель, соответствующая чекпоинту
        # 2. заменить в текущем конфиге весь раздел cfg.model
        # 3. создать модель
        # 4. залить веса
        # 5. перезаписать .hydra/config.yaml обновлённым конфигом, чтоб потом можно было инферить
        logger.info(f"creating model for {cfg.checkpoint_path}")
        train_cfg = OmegaConf.load(os.path.join(os.path.dirname(cfg.checkpoint_path), "..", ".hydra", "config.yaml"))
        model = hydra.utils.instantiate(train_cfg.model)
        logger.info(f"loading weights")
        model.load_state_dict(torch.load(cfg.checkpoint_path, map_location="cpu"))
        logger.info(f"overriding existing config")
        cfg.model = train_cfg.model
        OmegaConf.save(cfg, os.path.join(args.output_dir, ".hydra", "config.yaml"))
    else:
        logger.info("creating model")
        model = hydra.utils.instantiate(cfg.model)

    trainer_cls = hydra.utils.instantiate(cfg.trainer_cls)
    trainer = trainer_cls(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        **cfg.trainer_params
    )
    if hasattr(trainer, "remove_callback"):
        trainer.remove_callback(MLflowCallback)
    if hasattr(trainer, "add_callback"):
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=10))

    trainer.train()


if __name__ == "__main__":
    main()
