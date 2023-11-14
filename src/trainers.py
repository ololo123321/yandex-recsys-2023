import os
from typing import Dict
from datetime import datetime
from itertools import cycle
import time
import glob
import shutil
import random
from collections import defaultdict
import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel
import numpy as np

import transformers
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer import TRAINER_STATE_NAME

from sklearn.metrics import average_precision_score

from src.custom_loss import ResampleLoss


class MySamplerV1(Sampler):
    """
    в отличие от torch.utils.data.RandomSampler не нужно явно менять перестановку от эпохи к эпохе.
    """

    def __init__(self, n: int):
        super().__init__(data_source=None)
        self.n = n

    def __iter__(self):
        pi = torch.randperm(self.n)
        i = 0
        while True:
            yield pi[i].item()
            i += 1
            if i == self.n:
                pi = torch.randperm(self.n)
                i = 0


class MySamplerV2(Sampler):
    """
    рабоатет только с batch_size=num_classes
    """

    def __init__(self, tag2tracks, track2index):
        super().__init__(data_source=None)
        self.tag2tracks = tag2tracks
        self.track2index = track2index
        self.tags = list(tag2tracks.keys())

    def __iter__(self):
        while True:
            batch = set()
            for tag in random.sample(self.tags, len(self.tags)):
                for track in random.sample(self.tag2tracks[tag], len(self.tag2tracks[tag])):
                    if track not in batch:
                        batch.add(track)
                        break
            for track in batch:
                yield self.track2index[track]


class BaseTrainer(Trainer):
    def __init__(self, **kwargs):
        self.save_weights_only = kwargs.pop("save_weights_only", True)
        super().__init__(**kwargs)
        self.num_classes = self.data_collator.num_classes
        self.logger = transformers.logging.get_logger("trainer")

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        assert self.train_dataset is not None
        # return DataLoader(
        #     self.train_dataset,
        #     batch_size=self.args.per_device_train_batch_size,
        #     sampler=MySamplerV1(len(self.train_dataset)),
        #     collate_fn=self.data_collator,
        #     drop_last=True,
        #     num_workers=self.args.dataloader_num_workers,
        #     pin_memory=self.args.dataloader_pin_memory,
        # )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=None,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset=None) -> torch.utils.data.DataLoader:
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        assert ds is not None
        return DataLoader(
            ds,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _postprocess_metrics(self, metrics: Dict) -> Dict:
        # * add prefix
        # * make json-serializable
        metrics = {f"eval_{k}": float(v) for k, v in metrics.items()}

        # из родительского класса
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics

    def _save_checkpoint(self, model, trial, metrics=None):
        if self.save_weights_only:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self.args.output_dir
            self.store_flos()
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir)

            # Determine the new best metric / best model checkpoint
            if metrics is not None and self.args.metric_for_best_model is not None:
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                metric_value = metrics[metric_to_check]

                operator = np.greater if self.args.greater_is_better else np.less
                if (
                        self.state.best_metric is None
                        or self.state.best_model_checkpoint is None
                        or operator(metric_value, self.state.best_metric)
                ):
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir

            # Save the Trainer state
            if self.args.should_save:
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
        else:
            super()._save_checkpoint(model=model, trial=trial, metrics=metrics)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """
        В самом конце обучения может вылазить ZeroDivisionError при расчёте
        лосса для записи в лог. Внёс соответствующий фикс
        """
        if self.control.should_log:
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            # tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # в конце обучения коллективная коммуникация в этом месте вызывает зависание
            # TODO: разобраться
            tr_loss_scalar = tr_loss.item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            d = max(self.state.global_step - self._globalstep_last_logged, 1)  # my fix here
            logs["loss"] = round(tr_loss_scalar / d, 4)
            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                logs["learning_rate"] = self._get_learning_rate()
            else:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    logs["learning_rate"] = float(param_group['lr'])
                    break

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.

        добавил логирование, чтоб метрики попадали в файл с логами
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        if self.args.local_rank in [-1, 0]:
            # tensorboard.compat.tensorflow_stub.errors.AlreadyExistsError: Directory already exists
            self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
            self.logger.info(output)

    def store_flos(self):
        """
        наблюдал зависания из-за этой функции
        TODO: разобраться, в чём дело
        """
        pass


class TrainerV1(BaseTrainer):
    def __init__(self, **kwargs):
        pos_weight_path = kwargs.pop("pos_weight_path", None)
        super().__init__(**kwargs)
        self.pos_weight = None
        if pos_weight_path is not None:
            self.logger.info(f"loading pos_weight from {pos_weight_path}")
            self.pos_weight = torch.tensor(np.load(pos_weight_path), device=self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        logits = model(**inputs)  # [N, C]
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fn(logits, inputs["labels"])
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval") -> Dict[str, float]:
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        assert ds is not None
        self.model.eval()
        n = len(ds)
        loader = self.get_eval_dataloader(ds)
        y_pred = torch.zeros((n, self.num_classes))
        y_true = torch.zeros((n, self.num_classes), dtype=torch.long)
        loss = 0.0
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
        i = 0
        pbar = tqdm.tqdm(
            desc=f"rank: {self.args.local_rank}",
            total=n,
            position=max(0, self.args.local_rank),
            leave=True
        )
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                logits = self.model(**batch)  # [N, C]
                loss += loss_fn(logits, batch["labels"])
                j = min(i + self.args.per_device_eval_batch_size, n)
                y_pred[i:j] = logits.to("cpu")
                y_true[i:j] = batch["labels"].to("cpu")
                pbar.update(j - i)
                i = j
        pbar.close()
        # torch.cuda.empty_cache()  # TODO: по флагу
        if self.args.world_size > 1:
            self.logger.info("all_gather y_true and y_pred")
            y_true = self._all_gather(y_true.to(self.args.device)).to("cpu")
            y_pred = self._all_gather(y_pred.to(self.args.device)).to("cpu")

        metrics = {
            "map": average_precision_score(y_true.numpy(), y_pred.numpy()),
            "loss": loss / n / self.num_classes
        }
        metrics = self._postprocess_metrics(metrics)
        return metrics

    def _all_gather(self, x):
        xs = [torch.empty_like(x) for _ in range(self.args.world_size)]
        dist.all_gather(xs, x)
        x = torch.cat(xs, dim=0)
        return x


class TrainerV2(TrainerV1):
    """
    Focal loss.
    original implementation:
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
    """
    def __init__(self, **kwargs):
        self.gamma = kwargs.pop("gamma", 2.0)
        self.alpha = kwargs.pop("alpha", 0.25)
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        logits = model(**inputs)  # [N, C]
        y = inputs["labels"].float()
        p = torch.sigmoid(logits)  # [N, C]
        ce_loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")  # [N, C]
        p_t = p * y + (1 - p) * (1 - y)  # p, if y==1, else (1 - p)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * y + (1 - self.alpha) * (1 - y)
            loss = alpha_t * loss
        return loss.mean()


class TrainerV3(TrainerV1):
    """
    distribution-balanced loss
    implementation:
    https://www.kdnuggets.com/2023/03/multilabel-nlp-analysis-class-imbalance-loss-function-approaches.html
    """
    def __init__(self, **kwargs):
        class_freq = np.load(kwargs.pop("class_freq_path"))
        train_num = kwargs.pop("train_num")
        super().__init__(**kwargs)
        self.loss_fn = ResampleLoss(
            reweight_func="rebalance",
            loss_weight=1.0,
            focal=dict(focal=True, alpha=0.5, gamma=2),
            logit_reg=dict(init_bias=0.05, neg_scale=2.0),
            map_param=dict(alpha=0.1, beta=10.0, gamma=0.405),
            class_freq=class_freq,
            train_num=train_num,
            device=self.args.device
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        logits = model(**inputs)  # [N, C]
        return self.loss_fn(cls_score=logits, label=inputs["labels"])


class TrainerV4(TrainerV1):
    """
    v2 + ema
    работает корректно только при gradient_accumulation_steps=1
    """
    def __init__(self, **kwargs):
        self.gamma = kwargs.pop("gamma", 2.0)
        self.alpha = kwargs.pop("alpha", 0.25)
        super().__init__(**kwargs)

        def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
            decay = 0.999
            return averaged_model_parameter * decay + (1 - 0.999) * model_parameter

        self.ema = AveragedModel(self.model, avg_fn=avg_fn)

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     self.ema.update_parameters(self.model)
    #     return super().compute_loss(model, inputs, return_outputs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval") -> Dict[str, float]:
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        assert ds is not None
        self.model.eval()
        n = len(ds)
        loader = self.get_eval_dataloader(ds)
        y_pred = torch.zeros((n, self.num_classes))
        y_pred_ema = torch.zeros((n, self.num_classes))
        y_true = torch.zeros((n, self.num_classes), dtype=torch.long)
        loss = 0.0
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
        i = 0
        pbar = tqdm.tqdm(
            desc=f"rank: {self.args.local_rank}",
            total=n,
            position=max(0, self.args.local_rank),
            leave=True
        )
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                logits = self.model(**batch)  # [N, C]
                loss += loss_fn(logits, batch["labels"])
                j = min(i + self.args.per_device_eval_batch_size, n)
                y_pred[i:j] = logits.to("cpu")
                y_true[i:j] = batch["labels"].to("cpu")
                logits_ema = self.ema(**batch)  # [N, C]
                y_pred_ema[i:j] = logits_ema.to("cpu")
                pbar.update(j - i)
                i = j
        pbar.close()
        # torch.cuda.empty_cache()  # TODO: по флагу
        if self.args.world_size > 1:
            self.logger.info("all_gather y_true and y_pred")
            y_true = self._all_gather(y_true.to(self.args.device)).to("cpu")
            y_pred = self._all_gather(y_pred.to(self.args.device)).to("cpu")

        metrics = {
            "map": average_precision_score(y_true.numpy(), y_pred.numpy()),
            "map_ema": average_precision_score(y_true.numpy(), y_pred_ema.numpy()),
            "loss": loss / n / self.num_classes
        }
        metrics = self._postprocess_metrics(metrics)

        self.ema.update_parameters(self.model)
        return metrics


class TrainerV5(TrainerV1):
    """
    трейнер из трансформерс с ассиметричным фокал лоссом
    """
    def __init__(self, **kwargs):
        self.loss_fn = AsymmetricLoss(
            gamma_pos=kwargs.pop("gamma_pos", 0.0),
            gamma_neg=kwargs.pop("gamma_neg", 0.0),
            clip=kwargs.pop("clip", 0.0),
            reduction=kwargs.pop("reduction", "mean")
        )
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        logits = model(**inputs)  # [N, C]
        return self.loss_fn(logits, inputs["labels"])


class LossTypes:
    FOCAL = "focal"
    ASL = "asl"
    LOGSUMEXP = "logsumexp"


class TrainerCustom:
    """
    Кастомный train loop. Реализован частный случай:
    * world_size = 1 (использую только этот сетап, поэтому не хочу писать лишние ифы)
    * focal loss
    Написан только для reduce_lr_on_plateau
    """
    def __init__(
            self,
            model,
            args: TrainingArguments,
            data_collator,
            train_dataset,
            eval_dataset,

            gamma: float = 1.0,
            alpha: float = -1.0,
            max_epochs_wo_improvement: int = 20,
            reduce_lr_patience: int = 5,  # in epochs
            reduce_lr_factor: float = 0.5,
            sampler: str = "v1",
            loss_type: str = LossTypes.FOCAL,
            use_dataloader: bool = False,  # без него почему-то очень медленно, а с ним память течёт
            legacy_optimizer: bool = False,
            **kwargs
    ):
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.gamma = gamma
        self.alpha = alpha
        self.max_epochs_wo_improvement = max_epochs_wo_improvement
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.num_classes = self.data_collator.num_classes
        self.use_dataloader = use_dataloader
        self.legacy_optimizer = legacy_optimizer

        self.logger = transformers.logging.get_logger("trainer")

        self.logger.info(f"loss type: {loss_type}")
        if loss_type == LossTypes.ASL:
            self.loss_fn = AsymmetricLoss(**kwargs)
            self.logger.info("asymmetric focal loss will be used")
            self.logger.info(f"gamma_neg: {self.loss_fn.gamma_neg}")
            self.logger.info(f"gamma_pos: {self.loss_fn.gamma_pos}")
            self.logger.info(f"clip: {self.loss_fn.clip}")
        elif loss_type == LossTypes.FOCAL:
            self.loss_fn = self.focal_loss
        elif loss_type == LossTypes.LOGSUMEXP:
            self.loss_fn = MyLoss()
        else:
            raise NotImplementedError(f"unexpected loss: {loss_type}")

        assert sampler in {"v1", "v2"}, sampler
        if sampler == "v2":
            assert args.per_device_train_batch_size == self.num_classes, \
                "sampler v2 has been implemented for batch equal to num classes"
        self.sampler = sampler

        self.model.to(self.args.device)
        self.optimizer = self._get_optimizer(self.model)
        self.scaler = torch.cuda.amp.GradScaler()

        now = datetime.now()
        log_dir = os.path.join(args.output_dir, "runs", now.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(log_dir)  # будут созданы и output_dir, и log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self):
        """
        учим до того, пока не наступит одно из следующих условий:
        * пройдено max_steps шагов
        * валидационная метрика не увеличивается max_num_steps_wo_improvement шагов
        """
        if self.sampler == "v1":
            sampler = MySamplerV1(len(self.train_dataset))
        elif self.sampler == "v2":
            tag2tracks = defaultdict(list)
            track2index = {}
            for i, (track, _, tags) in enumerate(self.train_dataset.data):
                track2index[track] = i
                for tag in tags:
                    tag2tracks[tag].append(track)
            sampler = MySamplerV2(tag2tracks, track2index)
        else:
            raise NotImplementedError

        if self.use_dataloader:
            self.logger.info("using torch dataloader")
            batch_iter = self._batch_iter_with_dataloader()
        else:
            self.logger.info("no torch dataloader")
            batch_iter = self._batch_iter_wo_dataloader()

        global_batch_size = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
        eval_steps = len(self.train_dataset) // global_batch_size
        if self.args.max_steps > 0:
            num_steps = self.args.max_steps
        else:
            num_steps = (len(self.train_dataset) * self.args.num_train_epochs) // global_batch_size

        self.logger.info(f"train size: {len(self.train_dataset)}")
        self.logger.info(f"max num steps: {num_steps}")
        self.logger.info(f"batch size: {global_batch_size}")
        self.logger.info(f"num steps per epoch: {eval_steps}")
        self.logger.info(f"num params: {sum(p.numel() for p in self.model.parameters())}")
        self.logger.info(f"num trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        pbar = tqdm.tqdm(total=num_steps)
        local_step = 0
        global_step = 0
        epoch = 1
        best_score = -1.0
        loss_acc = 0.0  # accumulated
        lr = self.args.learning_rate
        num_epochs_wo_improvement = 0
        curr_checkpoints = []
        t0 = time.time()  # инициализируем время старта шага

        while global_step < num_steps:
            # accumulate gradients
            batch = {k: v.to(self.args.device) for k, v in next(batch_iter).items()}
            loss_acc += self.accumulate_gradients(batch)
            local_step += 1

            # try to update weights
            if local_step % self.args.gradient_accumulation_steps == 0:
                # 1. клипаем градиенты:
                # 1.1. делим градиенты на scale_factor
                self.scaler.unscale_(self.optimizer)
                # 1.2. клипаем градиенты
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                # 2. обновляем веса (maybe)
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scale_after = self.scaler.get_scale()
                optimizer_was_run = scale_before <= scale_after
                self.model.zero_grad()

                # 3. если веса были обновлены, то обновляем scheduler
                if optimizer_was_run:
                    # self.scheduler.step()
                    pass
                else:
                    self.logger.warning(f"[step {global_step}] step ignored. "
                                        f"scale_before: {scale_before}; "
                                        f"scale after: {scale_after}")

                # 4. логгируем разные метрики обучения
                global_step_time = time.time() - t0
                loss_mean = loss_acc / self.args.gradient_accumulation_steps
                if global_step % self.args.logging_steps == 0:
                    self.writer.add_scalar("train/loss", loss_mean, global_step)
                    self.writer.add_scalar("train/seconds_per_step", global_step_time, global_step)
                    self.writer.add_scalar(
                        "train/epoch", global_batch_size * global_step / len(self.train_dataset), global_step
                    )
                    self.writer.add_scalar("train/lr", lr, global_step)
                    self.logger.info(f"[step {global_step}] loss: {loss_mean}; cache size: {len(self.train_dataset.index2emb)}")

                loss_acc = 0.0
                local_step = 0
                global_step += 1
                pbar.update(1)

                # evaluate
                if (global_step % eval_steps == 0) and (self.eval_dataset is not None):
                    self.logger.info(f"[step {global_step}] evaluation starts")
                    metrics = self.evaluate(self.model)
                    self.logger.info(f'[epoch {epoch}] eval metrics: {metrics}')
                    self.writer.add_scalar("valid/map", metrics["map"], global_step)
                    self.writer.add_scalar("valid/loss", metrics["loss"], global_step)
                    epoch += 1

                    # maybe save checkpoint
                    curr_score = metrics["map"]
                    if curr_score > best_score:
                        self.logger.info(f"[step {global_step}] !!! best score changed: "
                                         f"{round(best_score, 4)} -> {round(curr_score, 4)}")
                        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        checkpoint_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
                        self.logger.info(f"[step {global_step}] saving checkpoint to {checkpoint_path}")
                        torch.save(self.model.state_dict(), checkpoint_path)
                        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
                        best_score = curr_score
                        curr_checkpoints.append((checkpoint_dir, curr_score))
                        num_epochs_wo_improvement = 0

                        # remove redundant checkpoints
                        if len(curr_checkpoints) > self.args.save_total_limit:
                            curr_checkpoints_sorted = sorted(curr_checkpoints, key=lambda x: x[1], reverse=True)
                            for checkpoint_dir, score in curr_checkpoints_sorted[self.args.save_total_limit:]:
                                self.logger.info(
                                    f"[step {global_step}] removing checkpoint {checkpoint_dir} with score {score} "
                                    f"due to save_total_limit={self.args.save_total_limit}"
                                )
                                shutil.rmtree(checkpoint_dir)
                            curr_checkpoints = curr_checkpoints_sorted[:self.args.save_total_limit]
                    else:
                        num_epochs_wo_improvement += 1
                        self.logger.info(f"current score: {curr_score}")
                        self.logger.info(f"best score: {best_score}")
                        self.logger.info(f"num epochs without improvement: {num_epochs_wo_improvement}")

                        # maybe stop training
                        if num_epochs_wo_improvement == self.max_epochs_wo_improvement:
                            self.logger.info(f"stop training due to {self.max_epochs_wo_improvement} "
                                             f"epochs wo improvement reached")
                            break

                        # maybe reduce lr
                        elif num_epochs_wo_improvement % self.reduce_lr_patience == 0:
                            checkpoint_dir, score = max(curr_checkpoints, key=lambda x: x[1])
                            checkpoint_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
                            self.logger.info(f"loading best model from {checkpoint_path} with score {score}")
                            self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
                            optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
                            self.logger.info(f"loading optimizer state from {optimizer_path}")
                            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
                            curr_lr = lr
                            lr *= self.reduce_lr_factor
                            self.logger.info(f"reducing lr: {curr_lr} -> {lr}")
                            for group in self.optimizer.param_groups:
                                group["lr"] = lr

                # обновляем время старта шага
                t0 = time.time()

        pbar.close()
        self.writer.close()

        if self.eval_dataset is None:
            checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
            self.logger.info(f"[step {global_step}] saving checkpoint to {checkpoint_path}")
            torch.save(self.model.state_dict(), checkpoint_path)
        else:
            self.logger.info("removing all optimizer states to save space")
            for path in tqdm.tqdm(glob.glob(os.path.join(self.args.output_dir, "checkpoint-*", "optimizer.pt"))):
                os.remove(path)

    def evaluate(self, model) -> Dict[str, float]:
        model.eval()
        n = len(self.eval_dataset)
        loader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        y_pred = torch.zeros((n, self.num_classes))
        y_true = torch.zeros((n, self.num_classes), dtype=torch.long)
        loss = 0.0
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
        i = 0
        pbar = tqdm.tqdm(
            desc=f"rank: {self.args.local_rank}",
            total=n,
            position=max(0, self.args.local_rank),
            leave=True
        )
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                logits = model(**batch)  # [N, C]
                loss += loss_fn(logits, batch["labels"]).item()
                j = min(i + self.args.per_device_eval_batch_size, n)
                y_pred[i:j] = logits.to("cpu")
                y_true[i:j] = batch["labels"].to("cpu")
                pbar.update(j - i)
                i = j
        pbar.close()
        # torch.cuda.empty_cache()  # TODO: по флагу

        metrics = {
            "map": average_precision_score(y_true.numpy(), y_pred.numpy()),
            "loss": loss / n / self.num_classes
        }
        return metrics

    def accumulate_gradients(self, batch):
        """
        1. compute loss
        2. divide by grad acc steps
        3. apply grad scaler
        4. accumulate grads
        """
        self.model.train()  # ensure model in training mode
        with torch.cuda.amp.autocast(cache_enabled=True, dtype=torch.float16):
            logits = self.model(**batch)
            loss = self.loss_fn(logits, batch["labels"])
        loss_value = loss.item()  # for logging
        loss = loss / self.args.gradient_accumulation_steps
        loss = self.scaler.scale(loss)
        loss.backward()
        return loss_value

    def focal_loss(self, logits, labels):
        y = labels.float()
        p = torch.sigmoid(logits)  # [N, C]
        ce_loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")  # [N, C]
        p_t = p * y + (1 - p) * (1 - y)  # p, if y==1, else (1 - p)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * y + (1 - self.alpha) * (1 - y)
            loss = alpha_t * loss
        return loss.mean()

    def _batch_iter_wo_dataloader(self):
        while True:
            idx = random.sample(range(len(self.train_dataset)), self.args.per_device_train_batch_size)
            yield self.data_collator([self.train_dataset[i] for i in idx])

    def _batch_iter_with_dataloader(self):
        """
        по идее можно сделать менее убого, если в DataLoader прокинуть торчовый
        RandomSampler с торчовым генератором, и если батч не получается вернуть,
        то просто у генератора поменять сид
        """
        it = None
        while True:
            # try to create batch from current iterator
            batch = None
            if it is not None:
                batch = next(it, None)

            # если удалось, то возвращаем его
            if batch is not None:
                yield batch
            # иначе - пересоздаём data loader
            else:
                it = iter(DataLoader(
                    dataset=self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    sampler=None,
                    shuffle=True,
                    collate_fn=self.data_collator,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory
                ))
                batch = next(it)
                yield batch

    def _get_optimizer(self, model):
        if self.legacy_optimizer:
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        self.logger.info(f"num params with decay: {len(decay)}")
        self.logger.info(f"num params wo decay: {len(no_decay)}")
        params = [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': self.args.weight_decay}
        ]
        # сделал фикс
        return torch.optim.AdamW(params=params, lr=self.args.learning_rate, weight_decay=0.0)


class TrainerCustomV2(TrainerCustom):
    """
    * lr обновляется по эпохам
    * fixed weight decay
    * он будет использован для обучения итоговых моделей на всём треине
    """
    def __init__(self, **kwargs):
        self.t_max = kwargs.pop("t_max", 100)
        self.min_lr = kwargs.pop("min_lr", 1e-7)
        self.use_ema = kwargs.pop("ema", False)
        super().__init__(**kwargs)
        self.ema = None
        if self.use_ema:
            self.ema = AveragedModel(self.model)

    # def _get_optimizer(self, model):
    #     decay = []
    #     no_decay = []
    #     for name, param in model.named_parameters():
    #         if not param.requires_grad:
    #             continue  # frozen weights
    #         if len(param.shape) == 1 or name.endswith(".bias"):
    #             no_decay.append(param)
    #         else:
    #             decay.append(param)
    #     self.logger.info(f"num params with decay: {len(decay)}")
    #     self.logger.info(f"num params wo decay: {len(no_decay)}")
    #     params = [
    #         {'params': no_decay, 'weight_decay': 0.},
    #         {'params': decay, 'weight_decay': self.args.weight_decay}
    #     ]
    #     return torch.optim.Adam(params=params, lr=self.args.learning_rate, weight_decay=0.0)

    def train(self):
        """
        учим до того, пока не наступит одно из следующих условий:
        * пройдено max_steps шагов
        * валидационная метрика не увеличивается max_num_steps_wo_improvement шагов
        """
        if self.sampler == "v1":
            sampler = MySamplerV1(len(self.train_dataset))
        elif self.sampler == "v2":
            tag2tracks = defaultdict(list)
            track2index = {}
            for i, (track, _, tags) in enumerate(self.train_dataset.data):
                track2index[track] = i
                for tag in tags:
                    tag2tracks[tag].append(track)
            sampler = MySamplerV2(tag2tracks, track2index)
        else:
            raise NotImplementedError

        if self.use_dataloader:
            self.logger.info("using torch dataloader")
            batch_iter = self._batch_iter_with_dataloader()
        else:
            self.logger.info("no torch dataloader")
            batch_iter = self._batch_iter_wo_dataloader()

        global_batch_size = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
        steps_per_epoch = len(self.train_dataset) // global_batch_size
        if self.args.max_steps > 0:
            num_steps = self.args.max_steps
        else:
            num_steps = steps_per_epoch * self.args.num_train_epochs

        self.logger.info(f"train size: {len(self.train_dataset)}")
        self.logger.info(f"max num steps: {num_steps}")
        self.logger.info(f"num epochs: {self.args.num_train_epochs}")
        self.logger.info(f"batch size: {global_batch_size}")
        self.logger.info(f"num steps per epoch: {steps_per_epoch}")
        self.logger.info(f"num trainable params: {sum(p.numel() for p in self.model.parameters())}")

        pbar = tqdm.tqdm(total=num_steps)
        local_step = 0
        global_step = 0
        epoch = 1
        best_score = -1.0
        loss_acc = 0.0  # accumulated
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.t_max, eta_min=self.min_lr)
        # scheduler = get_cosine_schedule_with_warmup(
        #     self.optimizer,
        #     num_warmup_steps=int(self.args.warmup_ratio * num_steps),
        #     num_training_steps=num_steps
        # )
        # def f(epoch):
        #     if epoch <= 25:
        #         return 1.0
        #     if epoch % 5 == 0:
        #         return 0.5
        #     else:
        #         return 1.0
        def f(epoch):
            if epoch <= 15:
                return 1.0
            return 0.90
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, f)
        num_epochs_wo_improvement = 0
        curr_checkpoints = []
        t0 = time.time()  # инициализируем время старта шага

        while global_step < num_steps:
            # accumulate gradients
            batch = {k: v.to(self.args.device) for k, v in next(batch_iter).items()}
            loss_acc += self.accumulate_gradients(batch)
            local_step += 1

            # try to update weights
            if local_step % self.args.gradient_accumulation_steps == 0:
                # 1. клипаем градиенты:
                # 1.1. делим градиенты на scale_factor
                self.scaler.unscale_(self.optimizer)
                # 1.2. клипаем градиенты
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                # 2. обновляем веса (maybe)
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scale_after = self.scaler.get_scale()
                optimizer_was_run = scale_before <= scale_after
                self.model.zero_grad()

                # 3. если веса были обновлены, то обновляем scheduler
                if optimizer_was_run:
                    # scheduler.step()
                    if self.ema is not None:
                        self.ema.update_parameters(self.model)
                    pass
                else:
                    self.logger.warning(f"[step {global_step}] step ignored. "
                                        f"scale_before: {scale_before}; "
                                        f"scale after: {scale_after}")

                # 4. логгируем разные метрики обучения
                global_step_time = time.time() - t0
                loss_mean = loss_acc / self.args.gradient_accumulation_steps
                if global_step % self.args.logging_steps == 0:
                    self.writer.add_scalar("train/loss", loss_mean, global_step)
                    self.writer.add_scalar("train/seconds_per_step", global_step_time, global_step)
                    self.writer.add_scalar(
                        "train/epoch", global_batch_size * global_step / len(self.train_dataset), global_step
                    )
                    self.writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                    self.logger.info(
                        f"[step {global_step}] loss: {loss_mean}; "
                        f"lr: {scheduler.get_last_lr()[0]}; cache size: {len(self.train_dataset.index2emb)}"
                    )

                # maybe update lr
                if global_step % steps_per_epoch == 0:
                    scheduler.step()  # this changed
                    epoch += 1

                loss_acc = 0.0
                local_step = 0
                global_step += 1
                pbar.update(1)

                # evaluate
                if (global_step % steps_per_epoch == 0) and (self.eval_dataset is not None):
                    self.logger.info(f"[step {global_step}] evaluation starts")
                    metrics = self.evaluate(self.model)
                    self.logger.info(f'[epoch {epoch}] eval metrics: {metrics}')
                    self.writer.add_scalar("valid/map", metrics["map"], global_step)
                    self.writer.add_scalar("valid/loss", metrics["loss"], global_step)
                    curr_score = metrics["map"]
                    if self.ema is not None:
                        metrics_ema = self.evaluate(self.ema)
                        self.logger.info(f'[epoch {epoch}] eval metrics ema: {metrics_ema}')
                        self.writer.add_scalar("valid_ema/map", metrics_ema["map"], global_step)
                        self.writer.add_scalar("valid_ema/loss", metrics_ema["loss"], global_step)
                        curr_score = metrics_ema["map"]

                    # maybe save checkpoint
                    if curr_score > best_score:
                        self.logger.info(f"[step {global_step}] !!! best score changed: "
                                         f"{round(best_score, 4)} -> {round(curr_score, 4)}")
                        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        checkpoint_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
                        self.logger.info(f"[step {global_step}] saving checkpoint to {checkpoint_path}")
                        m = self.ema if self.ema is not None else self.model
                        torch.save(m.state_dict(), checkpoint_path)
                        # torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
                        best_score = curr_score
                        curr_checkpoints.append((checkpoint_dir, curr_score))
                        num_epochs_wo_improvement = 0

                        # remove redundant checkpoints
                        if len(curr_checkpoints) > self.args.save_total_limit:
                            curr_checkpoints_sorted = sorted(curr_checkpoints, key=lambda x: x[1], reverse=True)
                            for checkpoint_dir, score in curr_checkpoints_sorted[self.args.save_total_limit:]:
                                self.logger.info(
                                    f"[step {global_step}] removing checkpoint {checkpoint_dir} with score {score} "
                                    f"due to save_total_limit={self.args.save_total_limit}"
                                )
                                shutil.rmtree(checkpoint_dir)
                            curr_checkpoints = curr_checkpoints_sorted[:self.args.save_total_limit]
                    else:
                        num_epochs_wo_improvement += 1
                        self.logger.info(f"current score: {curr_score}")
                        self.logger.info(f"best score: {best_score}")
                        self.logger.info(f"num epochs without improvement: {num_epochs_wo_improvement}")

                        # maybe stop training
                        if num_epochs_wo_improvement == self.max_epochs_wo_improvement:
                            self.logger.info(f"stop training due to {self.max_epochs_wo_improvement} "
                                             f"epochs wo improvement reached")
                            break

                # обновляем время старта шага
                t0 = time.time()

        pbar.close()
        self.writer.close()

        if self.eval_dataset is None:
            checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
            self.logger.info(f"saving checkpoint at last step to {checkpoint_path}")
            m = self.ema if self.ema is not None else self.model
            torch.save(m.state_dict(), checkpoint_path)

        # self.logger.info("removing all optimizer states to save space")
        # for path in tqdm.tqdm(glob.glob(os.path.join(self.args.output_dir, "checkpoint-*", "optimizer.pt"))):
        #     os.remove(path)


class AsymmetricLoss(torch.nn.Module):
    # https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, reduction="mean", **kwargs):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        # reduction
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise
        return -loss


class MyLoss(torch.nn.Module):
    """
    размазываем единицу по позитивным классам
    """
    def forward(self, x, y):
        logits_all = torch.logsumexp(x, dim=-1)  # [N]
        x_masked = x + (1.0 - y.float()) * -10000.0
        logits_pos = torch.logsumexp(x_masked, dim=-1)
        return (logits_all - logits_pos).mean()
