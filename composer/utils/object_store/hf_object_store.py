# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Hub object store."""

from __future__ import annotations

import os
import pathlib
import shutil
from typing import Callable, Optional, Union

from composer.utils.object_store.object_store import ObjectStore, ObjectStoreTransientError

__all__ = ['HFObjectStore']

# HTTP status codes that indicate transient errors worth retrying
_TRANSIENT_STATUS_CODES = (429, 500, 502, 503, 504)


class HFObjectStore(ObjectStore):
    """Upload to and download from a HuggingFace Hub repository.

    Uses the ``huggingface_hub`` library. Authenticate by setting the
    ``HF_TOKEN`` environment variable or by calling ``huggingface_hub.login()``.

    Args:
        bucket (str): The HuggingFace repo id, e.g. ``'myorg/my-checkpoints'``.
        prefix (str): Path prefix prepended to every object name. Defaults to ``''``.
        repo_type (str): One of ``'model'``, ``'dataset'``, or ``'space'``. Defaults to ``'model'``.
        token (str, optional): HuggingFace API token. Falls back to ``HF_TOKEN`` env var.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = '',
        repo_type: str = 'model',
        token: Optional[str] = None,
    ) -> None:
        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise ImportError(
                'huggingface_hub is required for HFObjectStore. '
                'Install it with: pip install huggingface_hub',
            ) from e

        self.repo_id = bucket
        self.prefix = prefix.strip('/')
        self.repo_type = repo_type
        self.token = token or os.environ.get('HF_TOKEN')
        self.api = HfApi(token=self.token)
        self.api.create_repo(self.repo_id, repo_type=self.repo_type, exist_ok=True)

    def _full_path(self, object_name: str) -> str:
        if self.prefix:
            return f'{self.prefix}/{object_name}'
        return object_name

    def get_uri(self, object_name: str) -> str:
        return f'hf://{self.repo_id}/{self._full_path(object_name)}'

    def upload_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        callback: Optional[Callable[[int, int], None]] = None,
        **kwargs,
    ) -> None:
        from huggingface_hub.errors import HfHubHTTPError
        try:
            self.api.upload_file(
                path_or_fileobj=str(filename),
                path_in_repo=self._full_path(object_name),
                repo_id=self.repo_id,
                repo_type=self.repo_type,
            )
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code in _TRANSIENT_STATUS_CODES:
                raise ObjectStoreTransientError(str(e)) from e
            raise

    def download_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        overwrite: bool = False,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        import huggingface_hub
        from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError

        filename = pathlib.Path(filename)
        if not overwrite and filename.exists():
            raise FileExistsError(f'{filename} already exists. Set overwrite=True to overwrite.')

        try:
            cached = huggingface_hub.hf_hub_download(
                repo_id=self.repo_id,
                filename=self._full_path(object_name),
                repo_type=self.repo_type,
                token=self.token,
            )
        except (EntryNotFoundError, RepositoryNotFoundError) as e:
            raise FileNotFoundError(f'Object {object_name!r} not found in {self.repo_id}') from e
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code in _TRANSIENT_STATUS_CODES:
                raise ObjectStoreTransientError(str(e)) from e
            raise

        filename.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, filename)

    def get_object_size(self, object_name: str) -> int:
        from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError
        try:
            infos = list(self.api.get_paths_info(
                self.repo_id,
                [self._full_path(object_name)],
                repo_type=self.repo_type,
            ))
        except (EntryNotFoundError, RepositoryNotFoundError) as e:
            raise FileNotFoundError(f'Object {object_name!r} not found in {self.repo_id}') from e
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code in _TRANSIENT_STATUS_CODES:
                raise ObjectStoreTransientError(str(e)) from e
            raise

        if not infos:
            raise FileNotFoundError(f'Object {object_name!r} not found in {self.repo_id}')
        return infos[0].size  # type: ignore[return-value]

    def list_objects(self, prefix: Optional[str] = None) -> list[str]:
        from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
        try:
            items = self.api.list_repo_tree(self.repo_id, recursive=True, repo_type=self.repo_type)
        except RepositoryNotFoundError as e:
            raise FileNotFoundError(f'Repo {self.repo_id!r} not found') from e
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code in _TRANSIENT_STATUS_CODES:
                raise ObjectStoreTransientError(str(e)) from e
            raise

        full_prefix = self._full_path(prefix) if prefix else self.prefix
        paths = []
        for item in items:
            p = item.path
            if full_prefix and not p.startswith(full_prefix):
                continue
            # Strip the store prefix so callers see object names, not full repo paths
            if self.prefix:
                p = p[len(self.prefix):].lstrip('/')
            paths.append(p)
        return paths
