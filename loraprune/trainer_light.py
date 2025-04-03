from transformers.trainer import (
    Trainer,
    TrainerState,
    TrainOutput,
    has_length,
    get_model_param_count,
    speed_metrics,
    TRAINER_STATE_NAME,
)
from transformers.trainer_callback import ExportableState
import loraprune.utils as utils
import math
import sys
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.utils import logging, is_torch_xla_available, is_apex_available
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
import os
from packaging import version
import shutil

if is_apex_available():
    from apex import amp

parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)
is_torch_less_than_1_11 = parsed_torch_version_base < version.parse("1.11")
logger = logging.get_logger(__name__)

class LoRAPruneTrainer(Trainer):
    def __init__(self, model,
                 train_dataset,
                 eval_dataset,
                 args,
                 data_collator,
                 ratio,
                 init_ratio,
                 warmup_iters,
                 cooldown_iters,
                 prune_freq,
                 prune_metric
                 ):
        super().__init__(model=model,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         args=args,
                         data_collator=data_collator
                         )
        self.ratio = ratio
        self.init_ratio = init_ratio
        self.warmup_iters = warmup_iters
        self.cooldown_iters = cooldown_iters
        self.prune_freq = prune_freq
        self.prune_metric = prune_metric

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use a single GPU or non-DP training."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        # Create optimizer and scheduler unconditionally for non-distributed training.
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Initialize trainer state without resuming from a checkpoint.
        self.state = TrainerState(
            stateful_callbacks=[cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)]
        )
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed.
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size:,}")
        logger.info(f"  Total train batch size (w. accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # NOTE: Resuming from checkpoints has been removed. Training always starts from scratch.
        # Update the callback handler references.
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            self.state.trial_name = self.hp_name(self._trial)

        self.state.trial_params = None
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0).to(args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip data skip logic since resuming is not supported.
        for _ in range(epochs_trained):
            for _ in train_dataloader:
                break

        total_batched_samples = 0
        if self.prune_metric == 'grad':
            utils.unfreeze(model)

        sensitivity_dict = utils.init_sensitivity_dict(model)
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader

            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            # Removed any RNG state loading from checkpoints.
            rng_to_sync = False
            steps_skipped = 0

            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if rng_to_sync:
                    rng_to_sync = False

                # Removed checkpoint skip logic.
                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                if total_batched_samples % args.gradient_accumulation_steps == 0 or (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                ):
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        if is_torch_xla_available():
                            grad_norm = None
                        elif is_apex_available() and hasattr(self.optimizer, "clip_grad_norm"):
                            grad_norm = self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            grad_norm = self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            grad_norm = model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    sensitivity_dict = utils.update_sensitivity_dict(model, sensitivity_dict, self.prune_metric)
                    ratio = utils.schedule_sparsity_ratio(
                        self.state.global_step, self.state.max_steps,
                        self.warmup_iters,
                        self.cooldown_iters, self.init_ratio, self.ratio
                    )

                    if (self.state.global_step) % self.prune_freq == 0 and ratio > self.init_ratio and ratio < self.ratio:
                        utils.local_prune(model, sensitivity_dict, ratio, self.ratio)

                    self.optimizer.step()
                    self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, grad_norm if grad_norm is not None else None, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, None, model, trial, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            self._load_best_model()

        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)
