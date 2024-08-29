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
@world_size(2)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_train(world_size: int):
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    model = SimpleModel().to("cuda")
    dataset = RandomClassificationDataset(size=8)
    dataloader = DataLoader(dataset, batch_size=2, sampler=dist.get_sampler(dataset))

    layer_plan = {'fc1': ColwiseParallel(), 'fc2': RowwiseParallel()}
    tp_config = {'layer_plan': layer_plan, 'tensor_parallel_degree': world_size}

    # init 1
    model.forward(dataloader[0])

    # init 2
    model.param_init_fn(model.fc1)
    model.param_init_fn(model.fc2)

    # init 3
    for name, param in model.named_modules():
        print(name)
        print(param)
        print()

    model.fc1.weight = torch.nn.init.normal_(model.fc1.weight)
    model.fc2.weight = torch.nn.init.normal_(model.fc2.weight)
    model.fc1.bias = torch.nn.init.normal_(model.fc1.bias)
    model.fc2.bias = torch.nn.init.normal_(model.fc2.bias)

    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        parallelism_config={'tp': tp_config},
        max_duration='3ba',
    )

    trainer.fit()

@pytest.mark.gpu
@world_size(4)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_fsdp_train(world_size: int):
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    model = SimpleModel()
    dataset = RandomClassificationDataset(size=8)
    dataloader = DataLoader(dataset, batch_size=2, sampler=dist.get_sampler(dataset))

    layer_plan = {'fc1': ColwiseParallel(), 'fc2': RowwiseParallel()}
    tp_config = {'layer_plan': layer_plan, 'tensor_parallel_degree': 2}

    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        parallelism_config={'tp': tp_config, 'fsdp': {}},
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

