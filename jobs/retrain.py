"""
обучение с нуля на старом конфиге.
по сути подгружается старый конфиг, в котором заменяются только пути к выборкам
"""
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


@hydra.main(config_path="../config", config_name="train_overrides")
def main(overrides: DictConfig):
    """
    обязательно:
    model_dir - новый параметр: путь к обученной модели, из которой надо взять конфиг
    embeddings_dir - он by design обязателен, хотя его можно было бы и не переопределять
    training_args.output_dir
    train_data_path - путь к новому train
    valid_data_path - путь к новому valid

    желательно:
    trainer_params.loss_type=asl  # может не быть в старом конфиге, но он во всех экспериментах фактически был таким
    trainer_params.reduction=sum  # аналогично
    trainer_params.legacy_optimizer=false  # в большинстве экспериментов было true, но с false правильно. это не критично влияет на качество
    trainer_params.use_dataloader=true  # пофиксил баг с утечкой памяти, поэтому можно так юзать

    не желательно:
    training_args.* (кроме output_dir)
    trainer.*
    trainer_params.(гиперпараметры лосса)

    нельзя:
    model.*
    """
    cfg = OmegaConf.load(os.path.join(overrides.model_dir, ".hydra", "config.yaml"))  # загрузим старый конфиг
    # TODO: эту же проверку надо сделать в bash файле запуска обучения!
    assert cfg.training_args.output_dir != overrides.training_args.output_dir, \
        f"please rename or remove output dir: {overrides.training_args.output_dir}"
    cfg = OmegaConf.merge(cfg, overrides)  # обновим старый конфиг
    # перезапишем конфиг, который был автоматически сгенерирован гидрой:
    OmegaConf.save(cfg, os.path.join(cfg.training_args.output_dir, ".hydra", "config.yaml"))
    print(OmegaConf.to_yaml(cfg))

    args = hydra.utils.instantiate(cfg.training_args)
    logger = get_logger(args)
    logger.info(f"world size: {args.world_size}")
    logger.info(f"output_dir: {args.output_dir}")
    logger.info(f"hydra outputs dir: {os.getcwd()}")

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

    logger.info("start train")
    trainer.train()


if __name__ == "__main__":
    main()
