# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor throughput during training."""
from __future__ import annotations

import warnings
from collections import deque
from typing import Any, Callable, Optional, Union

import torch

from composer.core import Callback, State
from composer.loggers import Logger
from composer.models.base import ComposerModel
from composer.utils import dist, is_xla_installed

if is_xla_installed():
    import torch_xla.core.xla_model as xm

__all__ = ['SpeedMonitor']

_HSERIES_SXM = {
    'fp64': 67e12,
    'fp32': 67e12,
    'tf32': 989e12 / 2,
    'fp16': 1.979e15 / 2,
    'amp_fp16': 1.979e15 / 2,
    'bf16': 1.979e15 / 2,
    'amp_bf16': 1.979e15 / 2,
    'fp8': 3.958e15 / 2,
    'amp_fp8': 3.958e15 / 2,
    'int8': 3.958e15 / 2,
}

GPU_AVAILABLE_FLOPS = {
    # source: https://resources.nvidia.com/en-us-data-center-overview-mc/en-us-data-center-overview/hpc-datasheet-sc23-h200
    'h200-sxm': _HSERIES_SXM,
    # source: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
    # nvidia publishes spec sheet with a 2x sparsity factor
    'h100-sxm': _HSERIES_SXM,
    'h100-pcie': {
        'fp64': 51e12,
        'fp32': 51e12,
        'tf32': 756e12 / 2,
        'fp16': 1.513e15 / 2,
        'amp_fp16': 1.513e15 / 2,
        'bf16': 1.513e15 / 2,
        'amp_bf16': 1.513e15 / 2,
        'fp8': 3.026e15 / 2,
        'amp_fp8': 3.026e15 / 2,
        'int8': 3.026e15 / 2,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    # sxm and pcie have same flop counts
    'a100': {
        'fp64': 19.5e12,
        'fp32': 19.5e12,
        'tf32': 156e12,
        'fp16': 312e12,
        'amp_fp16': 312e12,
        'bf16': 312e12,
        'amp_bf16': 312e12,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/a10-datasheet.pdf
    'a10': {
        'fp32': 31.2e12,
        'tf32': 62.5e12,
        'fp16': 125e12,
        'amp_fp16': 125e12,
        'bf16': 125e12,
        'amp_bf16': 125e12,
    },
    # source: https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf
    'v100-sxm': {
        'fp64': 7.8e12,
        'fp32': 15.7e12,
        'fp16': 125e12,
        'amp_fp16': 125e12,
    },
    'v100-pcie': {
        'fp64': 7e12,
        'fp32': 14e12,
        'fp16': 112e12,
        'amp_fp16': 112e12,
    },
    'v100s-pcie': {
        'fp64': 8.2e12,
        'fp32': 16.4e12,
        'fp16': 130e12,
        'amp_fp16': 130e12,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
    # sxm and pcie have same flop counts
    't4': {
        'fp32': 8.1e12,
        'fp16': 65e12,
        'amp_fp16': 65e12,
        'int8': 130e12,
        'int4': 260e12,
    },
    # source: https://aws.amazon.com/blogs/machine-learning/aws-inferentia2-builds-on-aws-inferentia1-by-delivering-4x-higher-throughput-and-10x-lower-latency/
    # Numbers are halved as the above flops is per chip and each chip appears as 2 devices.
    'trn1': {
        'fp32': 47.5e12 / 2,
        'tf32': 47.5e12 / 2,
        'fp16': 190e12 / 2,
        'amp_fp16': 190e12 / 2,
        'bf16': 190e12 / 2,
        'amp_bf16': 190e12 / 2,
        'int8': 380e12 / 2,
    },
}


def get_gpu_flops_available(state: State):
    gpu_flops_available = None
    print('running get_gpu_flops_available')

    # Return 0 if no CUDA device (e.g., when running with CPU only)
    if torch.cuda.is_available():
        # torch.cuda.get_device_name() ex output: 'NVIDIA A100-SXM4-40GB'
        device_name = torch.cuda.get_device_name().lower()
        print(f'{device_name=}')
        if 'h200' in device_name:
            # We just assume SXM because device name does not differentiate, and we would have to check
            # power or bandwidth or something.
            device_name = 'h200-sxm'
        elif 'h100' in device_name and 'hbm3' in device_name:
            device_name = 'h100-sxm'
        elif 'h100' in device_name and ('pcie' in device_name or 'hbm2e' in device_name):
            device_name = 'h100-pcie'
        elif 'a100' in device_name:
            device_name = 'a100'
        elif 'a10' in device_name:
            device_name = 'a10'
        elif 'v100-sxm' in device_name:
            device_name = 'v100-sxm'
        elif 'v100-pcie' in device_name:
            device_name = 'v100-pcie'
        elif 't4' in device_name:
            device_name = 't4'
    elif is_xla_installed():
        if xm.xla_device_hw(xm.xla_device()) == 'NEURON':
            device_name = 'trn1'
        else:
            # For TPU return 0
            return 0
    else:
        # When running on CPU, return 0 without warning
        return 0

    if device_name in GPU_AVAILABLE_FLOPS and state.precision.value in GPU_AVAILABLE_FLOPS[device_name]:
        gpu_flops_available = int(GPU_AVAILABLE_FLOPS[device_name][state.precision.value])
    else:
        gpu_flops_available = None

    if gpu_flops_available is None:
        warnings.warn(
            f'gpu_flop count not found for {device_name} with precision={state.precision.value} ' +\
            f'so MFU cannot be calculated and reported. gpu_flops_available can be manually ' +\
            f'overridden by setting gpu_flops_available in SpeedMonitor or {device_name} can ' +\
            f'be added to GPU_AVAILABLE_FLOPS in composer/callbacks/speed_monitor.py',
            stacklevel=2,
        )
        # Setting to 0 will disable MFU computation and prevent
        # the speed monitor from running this helper every batch
        gpu_flops_available = 0

    return gpu_flops_available


class SpeedMonitor(Callback):
    """Logs the training throughput and utilization.

    The training throughput is logged on the :attr:`.Event.BATCH_END` event once we have reached
    the `window_size` threshold. If a model has `flops_per_batch` attribute, then flops per second
    is also logged. If running on a known GPU type or if `gpu_flops_available` is set, then MFU is
    also logged. All metrics are also logged as per device by dividing by world size.

    To compute `flops_per_sec`, the model attribute `flops_per_batch` should be set to a callable
    which accepts a batch and returns the number of flops for that batch. Typically, this should
    be flops per sample times the batch size unless pad tokens are used.

    The wall clock time is logged on every :attr:`.Event.BATCH_END` event.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import SpeedMonitor
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration='1ep',
            ...     callbacks=[SpeedMonitor(window_size=100)],
            ... )

    The training throughput is logged by the :class:`.Logger` to the following keys as
    described below.

    +-------------------------------------+-----------------------------------------------------------+
    | Key                                 | Logged data                                               |
    +=====================================+===========================================================+
    |                                     | Rolling average (over `window_size` most recent           |
    | `throughput/batches_per_sec`        | batches) of the number of batches processed per second    |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | Rolling average (over `window_size` most recent           |
    | `throughput/samples_per_sec`        | batches) of the number of samples processed per second    |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | Rolling average (over `window_size` most recent           |
    | `throughput/tokens_per_sec`         | batches) of the number of tokens processed per second.    |
    |                                     | Only logged if dataspec returns tokens per batch          |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | Estimates flops by `flops_per_batch * batches_per_sec`    |
    | `throughput/flops_per_sec`          | if model has attribute `flops_per_batch`                  |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    | `throughput/device/batches_per_sec` | `throughput/batches_per_sec` divided by world size        |
    +-------------------------------------+-----------------------------------------------------------+
    | `throughput/device/samples_per_sec` | `throughput/samples_per_sec` divided by world size        |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | `throughput/tokens_per_sec` divided by world size. Only   |
    | `throughput/device/tokens_per_sec`  | logged if dataspec returns tokens per batch               |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | `throughput/flops_per_sec` divided by world size. Only    |
    | `throughput/device/flops_per_sec`   | logged when model has attribute `flops_per_batch`         |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | `throughput/device/flops_per_sec` divided by flops        |
    |                                     | available on the GPU device. Only logged when model has   |
    | `throughput/device/mfu`             | attribute `flops_per_batch` and `gpu_flops_available`,    |
    |                                     | which can be passed as an argument if not automatically   |
    |                                     | determined by SpeedMonitor                                |
    +-------------------------------------+-----------------------------------------------------------+
    | `time/train`                        | Total elapsed training time                               |
    +-------------------------------------+-----------------------------------------------------------+
    | `time/val`                          | Total elapsed validation time                             |
    +-------------------------------------+-----------------------------------------------------------+
    | `time/total`                        | Total elapsed time (time/train + time/val)                |
    +-------------------------------------+-----------------------------------------------------------+

    Args:
        window_size (int, optional): Number of batches to use for a rolling average of throughput.
            Defaults to 100.
        gpu_flops_available (float, optional): Number of flops available on the GPU. If not set,
            SpeedMonitor will attempt to determine this automatically. Defaults to None.
        time_unit (str, optional): Time unit to use for `time` logging. Can be one of
            'seconds', 'minutes', 'hours', or 'days'. Defaults to 'hours'.
    """

    def __init__(
        self,
        window_size: int = 100,
        gpu_flops_available: Optional[Union[float, int]] = None,
        time_unit: str = 'hours',
    ):
        # Track the batch num samples and wct to compute throughput over a window of batches
        # Track seperately for train, eval, and total
        self.train_history_samples: deque[int] = deque(maxlen=window_size + 1)
        self.train_history_tokens: deque[int] = deque(maxlen=window_size + 1)
        self.train_history_wct: deque[float] = deque(maxlen=window_size + 1)
        self.train_history_flops: deque[float] = deque(maxlen=window_size + 1)

        self.eval_history_samples: deque[int] = deque(maxlen=window_size + 1)
        self.eval_history_tokens: deque[int] = deque(maxlen=window_size + 1)
        self.eval_history_wct: deque[float] = deque(maxlen=window_size + 1)
        self.eval_history_flops: deque[float] = deque(maxlen=window_size + 1)

        self.total_history_samples: deque[int] = deque(maxlen=window_size + 1)
        self.total_history_tokens: deque[int] = deque(maxlen=window_size + 1)
        self.total_history_wct: deque[float] = deque(maxlen=window_size + 1)
        self.total_history_flops: deque[float] = deque(maxlen=window_size + 1)

        self.gpu_flops_available = gpu_flops_available

        self.divider = 1
        if time_unit == 'seconds':
            self.divider = 1
        elif time_unit == 'minutes':
            self.divider = 60
        elif time_unit == 'hours':
            self.divider = 60 * 60
        elif time_unit == 'days':
            self.divider = 60 * 60 * 24
        else:
            raise ValueError(
                f'Invalid time_unit: {time_unit}. Must be one of "seconds", "minutes", "hours", or "days".',
            )

        # eval_timestep is reset after every evaluation so we must track the cumulative statistics across all the evaluations
        self.cumulative_eval_samples = 0.0
        self.cumulative_eval_tokens = 0.0
        self.cumulative_eval_wct = 0.0

        # state does not explictly track flops so we do it here
        self.cumulative_train_flops = 0.0
        self.cumulative_eval_flops = 0.0
        self.cumulative_total_flops = 0.0

    def state_dict(self) -> dict[str, Any]:
        return {
            'cumulative_eval_samples': self.cumulative_eval_samples,
            'cumulative_eval_tokens': self.cumulative_eval_tokens,
            'cumulative_eval_wct': self.cumulative_eval_wct,
            'cumulative_train_flops': self.cumulative_train_flops,
            'cumulative_eval_flops': self.cumulative_eval_flops,
            'cumulative_total_flops': self.cumulative_total_flops,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.cumulative_eval_samples = state['cumulative_eval_samples']
        self.cumulative_eval_tokens = state['cumulative_eval_tokens']
        self.cumulative_eval_wct = state['cumulative_eval_wct']
        self.cumulative_train_flops = state['cumulative_train_flops']
        self.cumulative_eval_flops = state['cumulative_eval_flops']
        self.cumulative_total_flops = state['cumulative_total_flops']

    def init(self, state: State, logger: Logger) -> None:
        print('running SpeedMonitor init')
        del logger  # unused
        if self.gpu_flops_available is None:
            self.gpu_flops_available = get_gpu_flops_available(state)

    def _log_throughput(self, logger: Logger, history_samples: deque[int], history_tokens: deque[int], history_wct: deque[float], mode: str):
        if len(history_wct) == history_wct.maxlen:
            world_size = dist.get_world_size()
            elapsed_batches = len(history_samples) - 1
            elapsed_samples = int(history_samples[-1]) - int(history_samples[0])
            elapsed_tokens = int(history_tokens[-1]) - int(history_tokens[0])
            elapsed_wct = history_wct[-1] - history_wct[0]
            batches_per_sec = elapsed_batches / elapsed_wct
            samples_per_sec = elapsed_samples / elapsed_wct
            dev_batches_per_sec = batches_per_sec / world_size
            dev_samples_per_sec = samples_per_sec / world_size
            logger.log_metrics({
                f'throughput/{mode}/batches_per_sec': batches_per_sec,
                f'throughput/{mode}/samples_per_sec': samples_per_sec,
                f'throughput/{mode}/device/batches_per_sec': dev_batches_per_sec,
                f'throughput/{mode}/device/samples_per_sec': dev_samples_per_sec,
            })
            if elapsed_tokens > 0:
                tokens_per_sec = elapsed_tokens / elapsed_wct
                dev_tokens_per_sec = tokens_per_sec / world_size
                logger.log_metrics({
                    f'throughput/{mode}/tokens_per_sec': tokens_per_sec,
                    f'throughput/{mode}/device/tokens_per_sec': dev_tokens_per_sec,
                })

    def _get_flops_per_batch(self, state: State):
        # Compute flops stats if model has flops_per_batch
        composer_model = state.model
        if not isinstance(composer_model, ComposerModel):
            composer_model = composer_model.module  # Pass through DDP wrapping
        if hasattr(composer_model, 'flops_per_batch'):
            model_flops_per_batch = composer_model.flops_per_batch  # type: ignore
            if not isinstance(model_flops_per_batch, Callable):
                raise TypeError(
                    'flops_per_batch must a callable accepting a batch and '
                    f'returning an int or float. Instead, got {type(model_flops_per_batch)}.',
                )
            device_flops_per_batch = model_flops_per_batch(state.batch)

            # Sum flops across all ranks since each rank computes the flops for its own batch
            flops_per_batch_tensor = state.device.tensor_to_device(
                torch.tensor(device_flops_per_batch, dtype=torch.float),
            )
            dist.all_reduce(flops_per_batch_tensor, reduce_operation='SUM')
            flops_per_batch = flops_per_batch_tensor.item()

            return flops_per_batch

    def _log_flops(self, logger: Logger, history_flops: deque[float], history_wct: deque[float], mode:str):

        # Log the flops throughput
        if len(history_flops) == history_flops.maxlen:
            world_size = dist.get_world_size()
            elapsed_flops = history_flops[-1] - history_flops[0]
            elapsed_wct = history_wct[-1] - history_wct[0]
            flops_per_sec = elapsed_flops / elapsed_wct
            device_flops_per_sec = flops_per_sec / world_size
            logger.log_metrics({
                f'throughput/{mode}/flops_per_sec': flops_per_sec,
                f'throughput/{mode}/device/flops_per_sec': device_flops_per_sec,
            })
            if self.gpu_flops_available:
                mfu = device_flops_per_sec / self.gpu_flops_available
                logger.log_metrics({f'throughput/{mode}/device/mfu': mfu})


    def batch_end(self, state: State, logger: Logger):
        # Add the new element
        self.train_history_samples.append(state.timestamp.sample.value)
        self.train_history_tokens.append(state.timestamp.token.value)
        self.train_history_wct.append(state.timestamp.total_wct.total_seconds())

        # Update total history
        self.total_history_samples.append(state.timestamp.sample.value + state.eval_timestamp.sample.value + self.cumulative_eval_samples)
        self.total_history_tokens.append(state.timestamp.token.value + state.eval_timestamp.token.value + self.cumulative_eval_tokens)
        self.total_history_wct.append(state.timestamp.total_wct.total_seconds() + state.eval_timestamp.total_wct.total_seconds() + self.cumulative_eval_wct)

        # Log the throughput
        self._log_throughput(logger, self.train_history_samples, self.train_history_tokens, self.train_history_wct, mode='train')
        self._log_throughput(logger, self.total_history_samples, self.total_history_tokens, self.total_history_wct, mode='total')

        # Log the flops and MFU
        flops_per_batch = self._get_flops_per_batch(state)
        if flops_per_batch is not None:
            self.cumulative_train_flops += flops_per_batch
            self.cumulative_total_flops += flops_per_batch
            self.train_history_flops.append(self.cumulative_train_flops)
            self.total_history_flops.append(self.cumulative_total_flops)
            self._log_flops(logger, self.train_history_flops, self.train_history_wct, mode='train')
            self._log_flops(logger, self.total_history_flops, self.total_history_wct, mode='total')

        # Log the time
        # `state.timestamp` excludes any time spent in evaluation
        logger.log_metrics({
            'time/train': self.train_history_wct[-1] / self.divider,
            'time/total' :self.total_history_wct[-1] / self.divider,
        })

    def eval_batch_end(self, state: State, logger: Logger):
        # Add the new element
        self.eval_history_samples.append(state.eval_timestamp.sample.value + self.cumulative_eval_samples)
        self.eval_history_tokens.append(state.eval_timestamp.token.value + self.cumulative_eval_tokens)
        self.eval_history_wct.append(state.eval_timestamp.total_wct.total_seconds() + self.cumulative_eval_wct)

        # Update total history
        self.total_history_samples.append(state.timestamp.sample.value + state.eval_timestamp.sample.value + self.cumulative_eval_samples)
        self.total_history_tokens.append(state.timestamp.token.value + state.eval_timestamp.token.value + self.cumulative_eval_tokens)
        self.total_history_wct.append(state.timestamp.total_wct.total_seconds() + state.eval_timestamp.total_wct.total_seconds() + self.cumulative_eval_wct)

        # Log the throughput
        self._log_throughput(logger, self.eval_history_samples, self.eval_history_tokens, self.eval_history_wct, mode='eval')
        self._log_throughput(logger, self.total_history_samples, self.total_history_tokens, self.total_history_wct, mode='total')

        # Log the flops and MFU
        flops_per_batch = self._get_flops_per_batch(state)
        if flops_per_batch is not None:
            self.cumulative_eval_flops += flops_per_batch
            self.cumulative_total_flops += flops_per_batch
            self.eval_history_flops.append(self.cumulative_eval_flops)
            self.total_history_flops.append(self.cumulative_total_flops)
            self._log_flops(logger, self.eval_history_flops, self.eval_history_wct, mode='eval')
            self._log_flops(logger, self.total_history_flops, self.total_history_wct, mode='total')

        # Log the time
        # `state.timestamp` excludes any time spent in evaluation
        logger.log_metrics({
            'time/val': self.eval_history_wct[-1] / self.divider,
            'time/total': self.total_history_wct[-1] / self.divider,
        })


    def eval_end(self, state: State, logger: Logger):
        del logger  # unused
        self.cumulative_eval_samples += state.eval_timestamp.sample.value
        self.cumulative_eval_tokens += state.eval_timestamp.token.value
        self.cumulative_eval_wct += state.eval_timestamp.total_wct.total_seconds()
