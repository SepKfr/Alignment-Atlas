from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


@dataclass
class StorageSyncResult:
    ok: bool
    backend: str
    action: str
    detail: str
    elapsed_seconds: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "backend": self.backend,
            "action": self.action,
            "detail": self.detail,
            "elapsed_seconds": self.elapsed_seconds,
        }


class StorageBackend:
    backend_name = "local"

    def sync_down(self) -> StorageSyncResult:
        return StorageSyncResult(
            ok=True,
            backend=self.backend_name,
            action="sync_down",
            detail="No-op local backend",
            elapsed_seconds=0.0,
        )

    def sync_up(self, *, commit_message: str) -> StorageSyncResult:
        return StorageSyncResult(
            ok=True,
            backend=self.backend_name,
            action="sync_up",
            detail="No-op local backend",
            elapsed_seconds=0.0,
        )

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.backend_name, "mode": "local_noop"}


class HFDatasetStorageBackend(StorageBackend):
    backend_name = "hf_dataset"

    def __init__(
        self,
        *,
        repo_id: str,
        token: str,
        subdir: str = "state",
        branch: str = "main",
    ) -> None:
        self.repo_id = repo_id
        self.token = token
        self.subdir = subdir.strip("/") or "state"
        self.branch = branch

    def _copy_into_local_data(self, downloaded_repo: Path) -> None:
        state_data = downloaded_repo / self.subdir / "data"
        if not state_data.exists():
            return
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for child in state_data.iterdir():
            target = DATA_DIR / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child, target)

    def sync_down(self) -> StorageSyncResult:
        started = time.time()
        try:
            from huggingface_hub import snapshot_download

            with tempfile.TemporaryDirectory(prefix="atlas-hf-syncdown-") as tmp:
                local_repo = Path(
                    snapshot_download(
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        token=self.token,
                        revision=self.branch,
                        local_dir=tmp,
                        local_dir_use_symlinks=False,
                    )
                )
                self._copy_into_local_data(local_repo)
            return StorageSyncResult(
                ok=True,
                backend=self.backend_name,
                action="sync_down",
                detail=f"Downloaded dataset state from {self.repo_id}",
                elapsed_seconds=round(time.time() - started, 2),
            )
        except Exception as e:
            return StorageSyncResult(
                ok=False,
                backend=self.backend_name,
                action="sync_down",
                detail=f"Sync down failed: {e}",
                elapsed_seconds=round(time.time() - started, 2),
            )

    def sync_up(self, *, commit_message: str) -> StorageSyncResult:
        started = time.time()
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self.token)
            api.create_repo(
                repo_id=self.repo_id,
                repo_type="dataset",
                private=True,
                exist_ok=True,
            )
            api.upload_folder(
                repo_id=self.repo_id,
                repo_type="dataset",
                folder_path=str(DATA_DIR),
                path_in_repo=f"{self.subdir}/data",
                commit_message=commit_message,
            )
            return StorageSyncResult(
                ok=True,
                backend=self.backend_name,
                action="sync_up",
                detail=f"Uploaded data/ to dataset {self.repo_id}",
                elapsed_seconds=round(time.time() - started, 2),
            )
        except Exception as e:
            return StorageSyncResult(
                ok=False,
                backend=self.backend_name,
                action="sync_up",
                detail=f"Sync up failed: {e}",
                elapsed_seconds=round(time.time() - started, 2),
            )

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_name,
            "repo_id": self.repo_id,
            "subdir": self.subdir,
            "branch": self.branch,
            "token_present": bool(self.token),
        }


def build_storage_backend_from_env() -> StorageBackend:
    mode = os.environ.get("STORAGE_BACKEND", "local").strip().lower()
    if mode != "hf_dataset":
        return StorageBackend()
    repo_id = os.environ.get("HF_DATASET_REPO", "").strip()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not repo_id or not token:
        return StorageBackend()
    subdir = os.environ.get("HF_DATASET_SUBDIR", "state").strip() or "state"
    branch = os.environ.get("HF_DATASET_BRANCH", "main").strip() or "main"
    return HFDatasetStorageBackend(repo_id=repo_id, token=token, subdir=subdir, branch=branch)

