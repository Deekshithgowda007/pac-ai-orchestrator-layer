from __future__ import annotations

import logging
import os
import re
from typing import Iterable, Optional

log = logging.getLogger("ai_inference.huggingface")

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_repo_id(repo_id: str) -> str:
    return _SAFE_NAME.sub("-", repo_id.strip())


def _required_paths_exist(base_dir: str, required_paths: Iterable[str]) -> bool:
    return all(os.path.exists(os.path.join(base_dir, path)) for path in required_paths)


def resolve_hf_bundle_dir(
    *,
    local_dir: str,
    repo_id: Optional[str],
    required_paths: Iterable[str],
    cache_root: Optional[str] = None,
) -> str:
    local_dir = (local_dir or "").strip()
    repo_id = (repo_id or "").strip()
    required_paths = tuple(required_paths)

    if local_dir and os.path.isdir(local_dir) and _required_paths_exist(local_dir, required_paths):
        return local_dir

    if not repo_id:
        return local_dir

    token = os.getenv("HF_API_KEY", "").strip() or None
    cache_root = (cache_root or os.getenv("HF_BUNDLE_CACHE_DIR", "/opt/model_cache/huggingface")).strip()
    target_dir = os.path.join(cache_root, _sanitize_repo_id(repo_id))

    if os.path.isdir(target_dir) and _required_paths_exist(target_dir, required_paths):
        return target_dir

    try:
        from huggingface_hub import snapshot_download

        os.makedirs(cache_root, exist_ok=True)
        downloaded_dir = snapshot_download(
            repo_id=repo_id,
            token=token,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        if downloaded_dir and _required_paths_exist(downloaded_dir, required_paths):
            log.info("Resolved Hugging Face bundle %s into %s", repo_id, downloaded_dir)
            return downloaded_dir
        if _required_paths_exist(target_dir, required_paths):
            log.info("Resolved Hugging Face bundle %s into %s", repo_id, target_dir)
            return target_dir
    except Exception as exc:
        log.warning("Unable to download Hugging Face bundle %s: %s", repo_id, exc)

    return local_dir or target_dir
