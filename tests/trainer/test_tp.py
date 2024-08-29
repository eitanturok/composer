# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader

from composer.loggers import InMemoryLogger
from composer.trainer.trainer import Trainer
from composer.utils import dist
from tests.common import (
    RandomClassificationDataset,
    SimpleModel,
    SimpleComposerMLP,
    SimpleDataset,
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
@world_size(4)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_correctness(world_size: int):
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
    import icecream
    icecream.install()

    SEED = 42

    def _helper(num_features: int = 2048, num_classes: int = 10, batch_size: int = 8, num_samples: int = 1024):
        """Return a model and distributed dataloader.

        Args:
            num_features (int, optional): the number of features in a single example. Defaults to 2048.
            num_classes (int, optional): the number of classes. Defaults to 10.
            batch_size (int, optional):  number of examples in a single batch on a single GPU. Defaults to 2.
            num_samples (int, optional): the size of the entire dataset. Defaults to 32.
        """
        model = SimpleComposerMLP(num_features=num_features, device='cpu', num_classes=num_classes)
        dataset = RandomClassificationDataset(shape=(num_features,), num_classes=num_classes, size=num_samples) # X=(num_features,), y=scalar
        dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset), batch_size=batch_size) # X=(batch_size, num_features), y=(batch_size,)
        # Are the dataloader dimensions per GPU? Or is are these dimensions split across every GPU?
        # what is the differenc between
        # batch
        # minibatch in dataloader (https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders)
        return model, dataloader


    # forward pass with no FSDP and no TP (DDP is done by default)
    model, dataloader = _helper()

    trainer = Trainer(
        seed=SEED,
        device='gpu',
        model=model,
        max_duration='1ba',
        train_dataloader=dataloader,
        # callbacks=[MemoryMonitor()],
        loggers=[InMemoryLogger()],
        )
    trainer.fit()
    logged_data = trainer.logger.destinations[0].data
    loss = logged_data['loss/train/total']
    accuracy = logged_data['metrics/train/MulticlassAccuracy']
    ic(logged_data.keys())

    for k, v in logged_data.items():
        ic(k, v)
    # outputs = torch.stack(trainer.predict(dataloader))


#    # forward pass with FSDP and no TP
#    model_fsdp, dataloader_fsdp = _helper()
#    trainer_fsdp = Trainer(
#        seed=SEED,
#        model=model_fsdp,
#        parallelism_config={'fsdp': {}},
#        # callbacks=[MemoryMonitor()],
#        # loggers=[InMemoryLogger()],
#        )
#    outputs_fsdp = torch.stack(trainer_fsdp.predict(dataloader_fsdp))


#    # forward pass with FSDP and TP
#    layer_plan = {'fc1': ColwiseParallel(), 'fc2': RowwiseParallel()}
#    tp_config = {'layer_plan': layer_plan, 'tensor_parallel_degree': 2}
#    model_fsdp_tp, dataloader_fsdp_tp = _helper()
#    trainer_fsdp_tp = Trainer(
#        seed=SEED,
#        model=model_fsdp_tp,
#        max_duration='1ba',
#        train_dataloader=dataloader_fsdp_tp,
#        # callbacks=[MemoryMonitor()],
#        loggers=[InMemoryLogger()],
#        parallelism_config={'fsdp': {}, 'tp': tp_config},
#        )
#    trainer_fsdp_tp.fit()


#    # match shape
#    assert outputs.shape == outputs_fsdp.shape
#    assert outputs.shape == outputs_fsdp_tp.shape


#    # match elements
#    assert torch.allclose(outputs, outputs_fsdp)
#    assert torch.allclose(outputs, outputs_fsdp_tp)
