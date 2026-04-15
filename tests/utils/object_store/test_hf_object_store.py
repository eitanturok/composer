# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for HFObjectStore (mocked — no network required)."""

import pathlib
import shutil
from unittest.mock import MagicMock, patch

import pytest

from composer.utils.object_store import ObjectStoreTransientError
from composer.utils.object_store.hf_object_store import HFObjectStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_api():
    """Return a MagicMock that stands in for HfApi."""
    api = MagicMock()
    api.create_repo.return_value = None
    return api


@pytest.fixture
def store(mock_api):
    """HFObjectStore with HfApi replaced by mock_api."""
    with patch('composer.utils.object_store.hf_object_store.HFObjectStore.__init__', lambda self, **kw: None):
        s = HFObjectStore.__new__(HFObjectStore)
    s.repo_id = 'myorg/my-checkpoints'
    s.prefix = ''
    s.repo_type = 'model'
    s.token = 'hf_fake'
    s.api = mock_api
    return s


@pytest.fixture
def store_with_prefix(mock_api):
    """HFObjectStore with a non-empty prefix."""
    with patch('composer.utils.object_store.hf_object_store.HFObjectStore.__init__', lambda self, **kw: None):
        s = HFObjectStore.__new__(HFObjectStore)
    s.repo_id = 'myorg/my-checkpoints'
    s.prefix = 'runs/exp1'
    s.repo_type = 'model'
    s.token = 'hf_fake'
    s.api = mock_api
    return s


# ---------------------------------------------------------------------------
# get_uri
# ---------------------------------------------------------------------------

def test_get_uri_no_prefix(store):
    assert store.get_uri('ep1.pt') == 'hf://myorg/my-checkpoints/ep1.pt'


def test_get_uri_with_prefix(store_with_prefix):
    assert store_with_prefix.get_uri('ep1.pt') == 'hf://myorg/my-checkpoints/runs/exp1/ep1.pt'


# ---------------------------------------------------------------------------
# upload_object
# ---------------------------------------------------------------------------

def test_upload_object(store, tmp_path):
    f = tmp_path / 'ckpt.pt'
    f.write_bytes(b'weights')

    store.upload_object('ep1/ckpt.pt', f)

    store.api.upload_file.assert_called_once_with(
        path_or_fileobj=str(f),
        path_in_repo='ep1/ckpt.pt',
        repo_id='myorg/my-checkpoints',
        repo_type='model',
    )


def test_upload_object_with_prefix(store_with_prefix, tmp_path):
    f = tmp_path / 'ckpt.pt'
    f.write_bytes(b'weights')

    store_with_prefix.upload_object('ep1/ckpt.pt', f)

    store_with_prefix.api.upload_file.assert_called_once_with(
        path_or_fileobj=str(f),
        path_in_repo='runs/exp1/ep1/ckpt.pt',
        repo_id='myorg/my-checkpoints',
        repo_type='model',
    )


def test_upload_object_transient_error(store, tmp_path):
    from huggingface_hub.errors import HfHubHTTPError

    f = tmp_path / 'ckpt.pt'
    f.write_bytes(b'x')

    resp = MagicMock()
    resp.status_code = 429
    err = HfHubHTTPError('rate limited', response=resp)
    store.api.upload_file.side_effect = err

    with pytest.raises(ObjectStoreTransientError):
        store.upload_object('ep1/ckpt.pt', f)


# ---------------------------------------------------------------------------
# download_object
# ---------------------------------------------------------------------------

def test_download_object(store, tmp_path):
    src = tmp_path / 'cached.pt'
    src.write_bytes(b'model weights')
    dest = tmp_path / 'out' / 'ckpt.pt'

    with patch('huggingface_hub.hf_hub_download', return_value=str(src)):
        store.download_object('ep1/ckpt.pt', dest)

    assert dest.read_bytes() == b'model weights'


def test_download_object_raises_file_exists(store, tmp_path):
    dest = tmp_path / 'ckpt.pt'
    dest.write_bytes(b'existing')

    with pytest.raises(FileExistsError):
        store.download_object('ep1/ckpt.pt', dest, overwrite=False)


def test_download_object_overwrite(store, tmp_path):
    src = tmp_path / 'cached.pt'
    src.write_bytes(b'new weights')
    dest = tmp_path / 'ckpt.pt'
    dest.write_bytes(b'old weights')

    with patch('huggingface_hub.hf_hub_download', return_value=str(src)):
        store.download_object('ep1/ckpt.pt', dest, overwrite=True)

    assert dest.read_bytes() == b'new weights'


def test_download_object_not_found(store, tmp_path):
    from huggingface_hub.errors import EntryNotFoundError

    dest = tmp_path / 'ckpt.pt'
    # EntryNotFoundError takes no keyword arguments — construct via MagicMock
    err = MagicMock(spec=EntryNotFoundError)
    with patch('huggingface_hub.hf_hub_download', side_effect=EntryNotFoundError):
        with pytest.raises(FileNotFoundError):
            store.download_object('missing.pt', dest)


# ---------------------------------------------------------------------------
# get_object_size
# ---------------------------------------------------------------------------

def test_get_object_size(store):
    info = MagicMock()
    info.size = 1234
    store.api.get_paths_info.return_value = iter([info])

    assert store.get_object_size('ep1/ckpt.pt') == 1234
    store.api.get_paths_info.assert_called_once_with(
        'myorg/my-checkpoints', ['ep1/ckpt.pt'], repo_type='model',
    )


def test_get_object_size_not_found(store):
    store.api.get_paths_info.return_value = iter([])

    with pytest.raises(FileNotFoundError):
        store.get_object_size('missing.pt')


# ---------------------------------------------------------------------------
# list_objects
# ---------------------------------------------------------------------------

def _make_item(path):
    item = MagicMock()
    item.path = path
    return item


def test_list_objects_no_prefix(store):
    store.api.list_repo_tree.return_value = iter([
        _make_item('ep1/ckpt.pt'),
        _make_item('ep2/ckpt.pt'),
    ])

    assert store.list_objects() == ['ep1/ckpt.pt', 'ep2/ckpt.pt']


def test_list_objects_with_prefix_arg(store):
    store.api.list_repo_tree.return_value = iter([
        _make_item('ep1/ckpt.pt'),
        _make_item('ep2/ckpt.pt'),
        _make_item('other/file.txt'),
    ])

    assert store.list_objects(prefix='ep') == ['ep1/ckpt.pt', 'ep2/ckpt.pt']


def test_list_objects_strips_store_prefix(store_with_prefix):
    store_with_prefix.api.list_repo_tree.return_value = iter([
        _make_item('runs/exp1/ep1/ckpt.pt'),
        _make_item('runs/exp1/ep2/ckpt.pt'),
        _make_item('runs/other/ckpt.pt'),
    ])

    result = store_with_prefix.list_objects()
    assert result == ['ep1/ckpt.pt', 'ep2/ckpt.pt']
