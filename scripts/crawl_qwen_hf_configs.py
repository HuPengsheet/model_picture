#!/usr/bin/env python3
"""Download config.json files for the canonical Qwen text-model families.

This intentionally excludes VL, Audio, Coder, Math, embedding/reranker models
and repackagings such as AWQ, GPTQ, GGUF and MLX.  Those are distinct product
families or quantized exports, rather than the seven families requested here.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# Base and instruction/chat checkpoints are both included when Qwen published
# both.  "Base" is explicit for Qwen3+; the short Qwen3 names are the released
# instruction checkpoints.
FAMILIES: dict[str, list[str]] = {
    "Qwen": ["Qwen-7B", "Qwen-14B", "Qwen-72B"],
    "Qwen1.5": [
        *(f"Qwen1.5-{size}{suffix}" for size in ("0.5B", "1.8B", "4B", "7B", "14B", "32B", "72B", "110B") for suffix in ("", "-Chat")),
        "Qwen1.5-MoE-A2.7B", "Qwen1.5-MoE-A2.7B-Chat",
    ],
    "Qwen2": [
        *(f"Qwen2-{size}{suffix}" for size in ("0.5B", "1.5B", "7B", "72B") for suffix in ("", "-Instruct")),
        "Qwen2-57B-A14B", "Qwen2-57B-A14B-Instruct",
    ],
    "Qwen2.5": [
        *(f"Qwen2.5-{size}{suffix}" for size in ("0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B") for suffix in ("", "-Instruct")),
    ],
    "Qwen3": [
        *(f"Qwen3-{size}" for size in ("0.6B", "1.7B", "4B", "8B", "14B", "32B", "30B-A3B", "235B-A22B")),
        *(f"Qwen3-{size}-Base" for size in ("0.6B", "8B", "14B", "30B-A3B", "235B-A22B")),
    ],
    "Qwen3.5": [
        "Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B",
        "Qwen3.5-35B-A3B", "Qwen3.5-122B-A10B", "Qwen3.5-397B-A17B",
        "Qwen3.5-0.8B-Base", "Qwen3.5-2B-Base", "Qwen3.5-4B-Base", "Qwen3.5-9B-Base",
    ],
}


def fetch(repo: str, family: str) -> dict:
    url = f"https://huggingface.co/Qwen/{repo}/resolve/main/config.json"
    request = Request(url, headers={"User-Agent": "model-picture-qwen-config-crawler/1.0"})
    last_error = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=45) as response:
                raw = response.read()
            config = json.loads(raw)
            return {
                "family": family,
                "repo": f"Qwen/{repo}",
                "url": url,
                "config": config,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1 + attempt)
    return {"family": family, "repo": f"Qwen/{repo}", "url": url, "error": last_error}


def summary(record: dict) -> dict:
    if "error" in record:
        return {key: record[key] for key in ("family", "repo", "url", "error")}
    c = record["config"]
    keys = ("model_type", "architectures", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size", "max_position_embeddings", "torch_dtype", "tie_word_embeddings")
    result = {key: record[key] for key in ("family", "repo", "url", "sha256")}
    result.update({key: c[key] for key in keys if key in c})
    # Qwen3.5's native multimodal checkpoints keep LLM dimensions in
    # text_config. Keep the top-level model type and expose those dimensions
    # with a prefix instead of silently reporting them as absent.
    if isinstance(c.get("text_config"), dict):
        text_config = c["text_config"]
        result.update({f"text_{key}": text_config[key] for key in keys if key in text_config})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/data/hf_qwen_configs"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = args.output
    jobs = [(repo, family) for family, repos in FAMILIES.items() for repo in repos]
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(fetch, repo, family): (repo, family) for repo, family in jobs}
        for future in concurrent.futures.as_completed(future_map):
            results.append(future.result())
    results.sort(key=lambda item: (item["family"], item["repo"]))

    for record in results:
        if "config" not in record:
            continue
        path = output / record["family"] / f"{record['repo'].split('/', 1)[1]}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record["config"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "source": "https://huggingface.co/Qwen",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Canonical text checkpoints in Qwen, Qwen1.5, Qwen2, Qwen2.5, Qwen3, and Qwen3.5; excludes modality-specific and quantized exports.",
        "models": [summary(record) for record in results],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failed = [item for item in results if "error" in item]
    print(f"Downloaded {len(results) - len(failed)}/{len(results)} configs to {output}")
    for item in failed:
        print(f"FAILED {item['repo']}: {item['error']}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
