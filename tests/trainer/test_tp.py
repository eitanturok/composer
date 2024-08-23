# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader

from composer.trainer.trainer import Trainer
from composer.utils import dist
from composer.callbacks import MemoryMonitor
from composer.loggers import InMemoryLogger
from tests.common import (
    RandomClassificationDataset,
    SimpleComposerMLP,
    SimpleDataset,
    SimpleModel,
    world_size,
)


@pytest.mark.gpu
@world_size(4)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_train(world_size: int):
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    model = SimpleModel()
    dataset = RandomClassificationDataset(size=8)
    dataloader = DataLoader(dataset, batch_size=2, sampler=dist.get_sampler(dataset))

    layer_plan = {
        'fc1': ColwiseParallel(),
        'fc2': RowwiseParallel(),
    }

    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        parallelism_config={
            'tp': {
                'layer_plan': layer_plan,
                'tensor_parallel_degree': 2,
            },
            'fsdp': {},
        },
        max_duration='3ba',
    )

    trainer.fit()


@pytest.mark.gpu
@world_size(4)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_with_param_groups(world_size: int):
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    model = SimpleModel()
    dataset = RandomClassificationDataset(size=8)
    dataloader = DataLoader(dataset, batch_size=2, sampler=dist.get_sampler(dataset))
    optimizer = torch.optim.SGD([{
        'params': model.fc1.parameters(),
        'lr': 0.1,
    }, {
        'params': model.fc2.parameters(),
        'lr': 0.5,
    }])

    layer_plan = {
        'fc1': ColwiseParallel(),
        'fc2': RowwiseParallel(),
    }

    expected_error = 'Multiple optimizer groups are not supported with tensor parallelism.'

    with pytest.raises(RuntimeError, match=expected_error):
        _ = Trainer(
            model=model,
            optimizers=optimizer,
            train_dataloader=dataloader,
            parallelism_config={
                'tp': {
                    'layer_plan': layer_plan,
                    'tensor_parallel_degree': 2,
                },
                'fsdp': {},
            },
            max_duration='3ba',
        )


@pytest.mark.gpu
@world_size(4)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_with_subset_of_params(world_size: int):
    from torch.distributed.tensor.parallel import ColwiseParallel

    model = SimpleModel()
    dataset = RandomClassificationDataset(size=8)
    dataloader = DataLoader(dataset, batch_size=2, sampler=dist.get_sampler(dataset))
    optimizer = torch.optim.SGD(model.fc1.parameters(), lr=0.1)

    layer_plan = {
        'fc1': ColwiseParallel(),
    }

    expected_error = 'Passing in a subset of model parameters to the optimizer is not supported with tensor parallelism.'

    with pytest.raises(ValueError, match=expected_error):
        _ = Trainer(
            model=model,
            optimizers=optimizer,
            train_dataloader=dataloader,
            parallelism_config={
                'tp': {
                    'layer_plan': layer_plan,
                    'tensor_parallel_degree': 2,
                },
                'fsdp': {},
            },
            max_duration='3ba',
        )


@pytest.mark.gpu
@world_size(8)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_correctness(world_size: int):
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
    from icecream import ic

    SEED = 42
    HIDDEN_DIM = 2048 # the number of features in a single example
    OUTPUT_DIM = 10 # the number of classes in a single example
    BATCH_DIM = 2 # number of examples in a single batch on a single GPU
    NUM_SAMPLES = 32 # the size of the entire dataset

    def _helper(hidden_dim: int, output_dim: int, batch_size: int, num_samples: int):
        model = SimpleComposerMLP(num_features=hidden_dim, device='cpu', num_classes=output_dim)
        dataset = SimpleDataset(size=num_samples, batch_size=batch_size, feature_size=hidden_dim, num_classes=output_dim)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=dist.get_sampler(dataset))
        return model, dataloader

    # forward pass with no FSDP and no TP (DDP is done by default)
    model, dataloader = _helper(hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, batch_size=BATCH_DIM, num_samples=NUM_SAMPLES)
    trainer = Trainer(
        seed=SEED,
        model=model
        # callbacks=[MemoryMonitor()],
        # loggers=[InMemoryLogger()],
        )
    outputs = torch.stack(trainer.predict(dataloader))

    # forward pass with FSDP and no TP
    model_fsdp, dataloader_fsdp = _helper(hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, batch_size=BATCH_DIM, num_samples=NUM_SAMPLES)
    trainer_fsdp = Trainer(
        seed=SEED,
        model=model_fsdp,
        parallelism_config={'fsdp': {}},
        # callbacks=[MemoryMonitor()],
        # loggers=[InMemoryLogger()],
        )
    outputs_fsdp = torch.stack(trainer_fsdp.predict(dataloader_fsdp))

    # forward pass with FSDP and TP
    layer_plan = {'fc1': ColwiseParallel(), 'fc2': RowwiseParallel()}
    tp_config = {'layer_plan': layer_plan, 'tensor_parallel_degree': 2}
    model_fsdp_tp, dataloader_fsdp_tp = _helper(hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, batch_size=BATCH_DIM, num_samples=NUM_SAMPLES)
    trainer_fsdp_tp = Trainer(
        seed=SEED,
        model=model_fsdp_tp,
        # callbacks=[MemoryMonitor()],
        # loggers=[InMemoryLogger()],
        parallelism_config={'fsdp': {}, 'tp': tp_config},
        )
    outputs_fsdp_tp = torch.stack(trainer_fsdp_tp.predict(dataloader_fsdp_tp))

    # match shape
    assert outputs.shape == outputs_fsdp.shape
    assert outputs.shape == outputs_fsdp_tp.shape

    # match elements
    assert torch.allclose(outputs, outputs_fsdp)
    assert torch.allclose(outputs, outputs_fsdp_tp)
