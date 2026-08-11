"""The per-dataset loaders. Download once into `experiments/data/`, then work offline.

Three access paths, by what upstream offers:

  * GitHub raw files (OpenAI moderation eval) — a single gzipped JSONL, no auth.
  * The Hugging Face datasets-server `rows` API (ToxicChat, BeaverTails, UnSmile,
    WildGuardMix) — paginated JSON, no client library needed. Gated datasets
    (WildGuardMix) send `HF_TOKEN` from the environment and refuse, with instructions,
    without it.
  * Manual placement (Jigsaw) — Kaggle terms require an account, so the loader reads a
    file the user downloads themselves and says exactly where to put it.

Every loader maps its labels to the one question this package asks — should this text
have been blocked? — and records how it did so. Where a dataset labels responses rather
than prompts (BeaverTails), the scored text is the prompt+response pair and the docstring
says so: pretending a response label describes its prompt would mislabel half the pool.
"""

import csv
import gzip
import io
import json
import os
import sys
import time
import random
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from guardopt.domain.types import ExpectedAction
from guardopt.runtime.materialise import LabelledRecord

__all__ = ["DATASETS", "DatasetSpec", "load_dataset"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

#: How many records a first fetch pulls into the cached pool. Sampling happens from the
#: pool, so this bounds download size, not experiment size — refetch with a higher cap
#: for the paper run.
DEFAULT_POOL_SIZE = 4_000

_HF_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}"
)


def _require(row: Mapping, field: str, dataset: str):
    if field not in row:
        raise ValueError(
            f"{dataset}: expected field '{field}' is missing — the upstream schema has "
            f"changed, and guessing a replacement would poison every label. Fields "
            f"present: {sorted(row)}"
        )
    return row[field]


def _http_bytes(url: str, *, headers: dict[str, str] | None = None) -> bytes:
    request = urllib.request.Request(url, headers=headers or {})
    # The datasets-server rate-limits a long paging run (HTTP 429), and a fetch that
    # dies two thirds through leaves a pool too small for the sample that was asked
    # for. Back off and retry rather than turning a transient limit into a failed
    # grid; a 5xx gets the same treatment, and anything else raises immediately
    # because retrying a 404 is just a slower 404.
    delay = 2.0
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            retryable = error.code == 429 or 500 <= error.code < 600
            if not retryable or attempt == 5:
                raise
            wait = float(error.headers.get("Retry-After") or delay)
            print(
                f"  datasets-server {error.code}; waiting {wait:.0f}s "
                f"(attempt {attempt + 1}/6)",
                file=sys.stderr,
            )
            time.sleep(wait)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable: the retry loop returns or raises")


def _hf_rows(
    dataset: str,
    config: str,
    split: str,
    pool_size: int,
    *,
    gated: bool = False,
) -> list[dict]:
    """Page through the datasets-server rows API, in order, until the pool is full.

    Sequential-from-zero is deliberate: the pool is deterministic given `pool_size`, and
    the stratified sample is seeded on top of it.
    """
    headers = {}
    if gated:
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            raise ValueError(
                f"{dataset} is gated on Hugging Face. Accept its licence on the hub, "
                f"then export HF_TOKEN (a read token). The loader sends it only to "
                f"huggingface.co."
            )
        headers["Authorization"] = f"Bearer {token}"

    rows: list[dict] = []
    offset = 0
    while len(rows) < pool_size:
        length = min(100, pool_size - len(rows))
        url = _HF_ROWS_URL.format(
            dataset=urllib.parse.quote(dataset, safe=""),
            config=config,
            split=split,
            offset=offset,
            length=length,
        )
        payload = json.loads(_http_bytes(url, headers=headers))
        page = [entry["row"] for entry in payload.get("rows", [])]
        if not page:
            break  # split exhausted
        rows.extend(page)
        offset += len(page)
        if len(rows) < pool_size:
            time.sleep(0.25)  # paced paging: cheaper than being rate-limited
    return rows


def _cache_path(name: str) -> Path:
    return DATA_DIR / f"{name}.pool.jsonl"


def _read_cache(name: str) -> list[LabelledRecord] | None:
    path = _cache_path(name)
    if not path.exists():
        return None
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        records.append(
            LabelledRecord(
                record_id=payload["id"],
                request={"text": payload["text"]},
                expected_action=ExpectedAction(payload["expected_action"]),
            )
        )
    return records


def _write_cache(name: str, records: list[LabelledRecord]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _cache_path(name).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "id": record.record_id,
                        "text": record.request["text"],
                        "expected_action": record.expected_action.value,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


# ──────────────────────────────────────────────────────────────────────────
# The datasets
# ──────────────────────────────────────────────────────────────────────────


def _openai_moderation(pool_size: int) -> list[LabelledRecord]:
    """OpenAI moderation eval set (openai/moderation-api-release, ~1.7k prompts).

    Labels: eight per-category flags; BLOCK when any is set. Licence: MIT (repo) —
    verify before redistribution.
    """
    raw = _http_bytes(
        "https://raw.githubusercontent.com/openai/moderation-api-release/main/data/"
        "samples-1680.jsonl.gz"
    )
    records = []
    with gzip.open(io.BytesIO(raw), "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= pool_size:
                break
            row = json.loads(line)
            text = str(_require(row, "prompt", "openai_moderation"))
            flags = [value for key, value in row.items() if key != "prompt"]
            unsafe = any(isinstance(v, (int, float)) and v >= 0.5 for v in flags)
            records.append(
                LabelledRecord(
                    record_id=f"omod_{index:05d}",
                    request={"text": text},
                    expected_action=ExpectedAction.BLOCK if unsafe else ExpectedAction.ALLOW,
                )
            )
    return records


def _toxicchat(pool_size: int) -> list[LabelledRecord]:
    """ToxicChat (lmsys/toxic-chat, toxicchat0124 test split): real user prompts.

    BLOCK when `toxicity` or `jailbreaking` is set — both are things a guardrail policy
    should stop. Licence: CC BY-NC 4.0 — non-commercial; verify before redistribution.
    """
    rows = _hf_rows("lmsys/toxic-chat", "toxicchat0124", "test", pool_size)
    records = []
    for index, row in enumerate(rows):
        text = str(_require(row, "user_input", "toxicchat"))
        unsafe = bool(_require(row, "toxicity", "toxicchat")) or bool(
            row.get("jailbreaking", 0)
        )
        records.append(
            LabelledRecord(
                record_id=f"tc_{index:05d}",
                request={"text": text},
                expected_action=ExpectedAction.BLOCK if unsafe else ExpectedAction.ALLOW,
            )
        )
    return records


def _beavertails(pool_size: int) -> list[LabelledRecord]:
    """BeaverTails (PKU-Alignment/BeaverTails, 30k test split).

    The label is QA-level (`is_safe` describes the prompt+response pair), so the scored
    TEXT is the pair too — labelling the prompt alone with a response verdict would
    mislabel every benign prompt that drew a harmful answer. Licence: CC BY-NC 4.0.
    """
    rows = _hf_rows("PKU-Alignment/BeaverTails", "default", "30k_test", pool_size)
    records = []
    for index, row in enumerate(rows):
        prompt = str(_require(row, "prompt", "beavertails"))
        response = str(_require(row, "response", "beavertails"))
        safe = bool(_require(row, "is_safe", "beavertails"))
        records.append(
            LabelledRecord(
                record_id=f"bt_{index:05d}",
                request={"text": f"{prompt}\n\n{response}"},
                expected_action=ExpectedAction.ALLOW if safe else ExpectedAction.BLOCK,
            )
        )
    return records


def _unsmile(pool_size: int) -> list[LabelledRecord]:
    """UnSmile Korean hate speech (smilegate-ai/kor_unsmile) — the WSDM'23 paper's set,
    kept for the head-to-head with TruSThresh. BLOCK when `clean` is 0.
    """
    rows = _hf_rows("smilegate-ai/kor_unsmile", "default", "valid", pool_size)
    records = []
    for index, row in enumerate(rows):
        text = str(_require(row, "문장", "unsmile"))
        clean = _require(row, "clean", "unsmile")
        records.append(
            LabelledRecord(
                record_id=f"us_{index:05d}",
                request={"text": text},
                expected_action=ExpectedAction.ALLOW if clean else ExpectedAction.BLOCK,
            )
        )
    return records


def _wildguardmix(pool_size: int) -> list[LabelledRecord]:
    """WildGuardMix test split (allenai/wildguardmix) — gated; needs HF_TOKEN.

    BLOCK when `prompt_harm_label` is 'harmful'. Licence: ODC-BY / AI2 terms — verify
    before redistribution.
    """
    rows = _hf_rows("allenai/wildguardmix", "wildguardtest", "test", pool_size, gated=True)
    records = []
    for index, row in enumerate(rows):
        text = str(_require(row, "prompt", "wildguardmix"))
        label = str(_require(row, "prompt_harm_label", "wildguardmix") or "")
        records.append(
            LabelledRecord(
                record_id=f"wg_{index:05d}",
                request={"text": text},
                expected_action=ExpectedAction.BLOCK
                if label == "harmful"
                else ExpectedAction.ALLOW,
            )
        )
    return records


def _jigsaw(pool_size: int) -> list[LabelledRecord]:
    """Jigsaw toxic comments — Kaggle terms require a logged-in download, so this reads
    `experiments/data/jigsaw/train.csv` placed there by hand. BLOCK when any of the six
    toxicity flags is set.
    """
    path = DATA_DIR / "jigsaw" / "train.csv"
    if not path.exists():
        raise ValueError(
            f"jigsaw needs a manual download (Kaggle account required): fetch "
            f"train.csv from the 'Toxic Comment Classification Challenge' and place it "
            f"at {path}. Kaggle's terms are why this one is not automated."
        )
    flags = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
    records = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index >= pool_size:
                break
            text = str(_require(row, "comment_text", "jigsaw"))
            unsafe = any(str(row.get(flag, "0")) == "1" for flag in flags)
            records.append(
                LabelledRecord(
                    record_id=f"jw_{index:05d}",
                    request={"text": text},
                    expected_action=ExpectedAction.BLOCK if unsafe else ExpectedAction.ALLOW,
                )
            )
    return records


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    fetch: Callable[[int], list[LabelledRecord]]
    licence_note: str


DATASETS: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in (
        DatasetSpec("openai_moderation", _openai_moderation, "MIT (verify)"),
        DatasetSpec("toxicchat", _toxicchat, "CC BY-NC 4.0 (verify)"),
        DatasetSpec("beavertails", _beavertails, "CC BY-NC 4.0 (verify)"),
        DatasetSpec("unsmile", _unsmile, "verify on the hub"),
        DatasetSpec("wildguardmix", _wildguardmix, "gated; ODC-BY/AI2 (verify)"),
        DatasetSpec("jigsaw", _jigsaw, "Kaggle terms; manual download"),
    )
}


def load_dataset(
    name: str,
    *,
    sample: int | None = None,
    seed: int = 0,
    pool_size: int = DEFAULT_POOL_SIZE,
    refresh: bool = False,
) -> list[LabelledRecord]:
    """Load a dataset's cached pool (fetching it on first use), optionally sampled.

    Sampling is stratified by expected action and seeded, so `sample=500, seed=0` names
    the same 500 cases forever — the property every table in the paper leans on. A
    requested sample larger than the pool is refused rather than silently shrunk.
    """
    spec = DATASETS.get(name)
    if spec is None:
        raise ValueError(f"unknown dataset {name!r}; known: {', '.join(sorted(DATASETS))}")

    records = None if refresh else _read_cache(name)
    if records is None:
        records = spec.fetch(pool_size)
        if not records:
            raise ValueError(f"{name}: the fetch returned no records")
        _write_cache(name, records)

    if sample is None or sample >= len(records):
        if sample is not None and sample > len(records):
            raise ValueError(
                f"{name}: asked for {sample} cases but the cached pool holds "
                f"{len(records)}. Refetch with a larger pool_size (refresh=True)."
            )
        return records

    rng = random.Random(seed)
    unsafe = [r for r in records if r.expected_action is ExpectedAction.BLOCK]
    safe = [r for r in records if r.expected_action is not ExpectedAction.BLOCK]
    unsafe_share = len(unsafe) / len(records)
    take_unsafe = min(len(unsafe), max(1, round(sample * unsafe_share)))
    take_safe = min(len(safe), sample - take_unsafe)
    chosen = rng.sample(unsafe, take_unsafe) + rng.sample(safe, take_safe)
    # Deterministic output order regardless of class interleaving.
    return sorted(chosen, key=lambda record: record.record_id)
