<div align="center">

# Beyond ICD Codes: Semantic IDs for Generative Medical Coding

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](#installation)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

<img src="assets/semicd_overview.png" alt="Overview of SemICD" width="85%">

</div>

SemICD treats ICD coding as generation. Each code becomes a hierarchical **Semantic ID (SID)**, and an LLM learns to write SIDs for a clinical note.

- **Semantic IDs.** Code profiles are embedded and clustered with hierarchical k-means, so similar codes share SID prefixes.
- **Bidirectional supervision.** Profile→SID and SID→profile tasks are trained alongside note→SID.
- **Constrained decoding.** Prefix-constrained beam search only produces valid SIDs, which are decoded to exact ICD sets.

## Installation

```bash
cd SemICD
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[train]'
```

Training needs Linux and an NVIDIA GPU with bfloat16 support; see the [Unsloth guide](https://unsloth.ai/docs/get-started/install) for a matching PyTorch build.

<details>
<summary>Quick check on synthetic data (no MIMIC or UMLS needed)</summary>

```bash
python examples/synthetic/make_model.py
semicd prepare-training --train-file examples/synthetic/train.jsonl \
  --profiles-file examples/synthetic/profiles.jsonl \
  --sid-bundle examples/synthetic/bundle --output-dir artifacts/tiny-training
semicd train --model-name-or-path "$PWD/artifacts/tiny-model" \
  --train-file artifacts/tiny-training/train.jsonl \
  --validation-file examples/synthetic/validation.jsonl \
  --max-seq-length 256 --max-new-tokens 12 --gradient-accumulation-steps 8 \
  --epochs 2 --learning-rate 0.01 --no-gradient-checkpointing \
  --output-dir artifacts/tiny-run
semicd evaluate --checkpoint artifacts/tiny-run \
  --eval-file examples/synthetic/validation.jsonl --output-dir artifacts/tiny-reloaded
```

To run the complete walkthrough and verify the saved checkpoint, use
`bash examples/synthetic/run.sh artifacts/synthetic-walkthrough` from an activated
training environment on an NVIDIA GPU. Use a fresh output directory each time.
The fixture is a plumbing check; its metrics do not measure medical coding quality.
The reused Python 3.12 training environment's package versions are recorded in
`examples/synthetic/constraints.txt`; the tested PyTorch build is `2.10.0+cu128`.
This version record is not a verification of a fresh dependency installation.

</details>

## Data

You need four inputs. ICD-9 Full uses diagnosis and procedure labels; ICD-10-CM uses diagnosis labels. MIMIC and UMLS must be obtained under their own access agreements.

| Input | Source |
| --- | --- |
| Notes | MIMIC-III v1.4 or MIMIC-IV v2.2, prepared with [Edin et al.](https://github.com/JoakimEdin/medical-coding-reproducibility) |
| Splits | Edin et al.'s released split table |
| ICD ontology | JSONL, one path per code |
| UMLS terms | JSON, code → list of terms |

> [!IMPORTANT]
> Use the **full diagnosis label set**: set `MIN_TARGET_COUNT = 1` in Edin et al.'s [`prepare_mimiciv.py`](https://github.com/JoakimEdin/medical-coding-reproducibility/blob/main/prepare_data/prepare_mimiciv.py).

<details>
<summary>Input formats</summary>

Codes are uppercase, with no decimal point and leading zeros kept. The notes table needs `_id`, `subject_id`, `text`, and `icd10_diag` (MIMIC-IV) or both `icd9_diag` and `icd9_proc` (ICD-9 Full / MIMIC-III). Splits need `_id` and `split` (`train`, `val`, `test`).

```json
{"code":"I10","code_system":"ICD-10-CM","edition":"FY2020","billable":true,"chapter":"Diseases of the circulatory system","block":"Hypertensive diseases","category":"Essential hypertension","description":"Essential (primary) hypertension"}
```

```json
{"I10": ["Essential (primary) hypertension", "Essential hypertension"]}
```

</details>

## Running the pipeline

Set the four input paths, then run the six steps. To generate the ICD-10-CM ontology and UMLS inputs after step 1, see the commands below:

```bash
NOTES=/path/to/mimiciv_icd10.feather
SPLITS=/path/to/mimiciv_icd10_split.feather
ONTOLOGY=/path/to/icd10cm_ontology.jsonl
UMLS=/path/to/icd10cm_umls_terms.json
OUT=artifacts/mimiciv

# 1. Diagnosis dataset (checked against the paper's Table 1)
semicd prepare-dataset --upstream-file $NOTES --splits-file $SPLITS --output-dir $OUT/data
# 2. Code profiles
semicd prepare-profiles --ontology-file $ONTOLOGY --umls-terms-file $UMLS \
  --codes-file $OUT/data/train_codes.json --output-dir $OUT/profiles
# 3. SID codebook (HKM, K=128, L=3, Qwen3-Embedding-4B)
semicd build-codebook --profiles-file $OUT/profiles/profiles.jsonl --output-dir $OUT/codebook
# 4. Training tasks: note→SID, profile→SID, SID→profile
semicd prepare-training --train-file $OUT/data/train.jsonl \
  --profiles-file $OUT/profiles/profiles.jsonl --sid-bundle $OUT/codebook --output-dir $OUT/tasks
# 5. Train; the best validation checkpoint is kept
semicd train --model-name-or-path Qwen/Qwen3-0.6B --train-file $OUT/tasks/train.jsonl \
  --validation-file $OUT/data/validation.jsonl --dataset-manifest $OUT/data/manifest.json \
  --output-dir $OUT/run
# 6. Evaluate on the test split
semicd evaluate --checkpoint $OUT/run --eval-file $OUT/data/test.jsonl --output-dir $OUT/test
```

Metrics are written to `$OUT/test/metrics.json`. For ICD-9 Full, prepare the mixed inputs below, then continue from step 3. The dataset is detected automatically. Use `Qwen/Qwen3-4B` for the larger backbone.

<details>
<summary>Baselines and ablations</summary>

Build a different codebook in step 3, then rerun steps 4–6 with it:

```bash
# Random SID: same SIDs, randomly reassigned to codes
semicd build-codebook --representation random --source-bundle $OUT/codebook --output-dir $OUT/random
# Atomic ID: one token per code
semicd build-codebook --representation atomic --codes-file $OUT/data/train_codes.json --output-dir $OUT/atomic
# RK-Means or RQ-VAE quantizer (reuses the step 3 embeddings)
semicd build-codebook --method rkmeans --profiles-file $OUT/profiles/profiles.jsonl \
  --embedding-dir $OUT/codebook-embeddings --output-dir $OUT/rkmeans
```

**Raw ICD** generates code strings directly and skips steps 2 and 4:

```bash
semicd build-codebook --representation raw --codes-file $OUT/data/train_codes.json --output-dir $OUT/raw
semicd train --model-name-or-path Qwen/Qwen3-0.6B --train-file $OUT/data/train.jsonl \
  --validation-file $OUT/data/validation.jsonl --sid-bundle $OUT/raw --output-dir $OUT/raw-run
```

</details>

### Preparing ICD-10-CM inputs

Download and extract the official CDC/NCHS FY2020 order TXT and tabular XML;
FY2015 is an optional fallback for codes absent from FY2020. Exact filenames and
source links are in [the source specification](configs/sources/icd10cm.json).
Use your own licensed UMLS `MRCONSO.RRF` and its actual release identifier:

```bash
# Run after prepare-dataset; both commands use the training code catalog.
semicd prepare-ontology --codes-file "$OUT/data/train_codes.json" \
  --primary-dir /path/to/extracted/FY2020 --supplement-dir /path/to/extracted/FY2015 \
  --output-dir "$OUT/ontology"
semicd prepare-umls --codes-file "$OUT/data/train_codes.json" \
  --umls-file /path/to/licensed/MRCONSO.RRF --release YOUR_ACTUAL_UMLS_RELEASE \
  --output-dir "$OUT/umls"
semicd prepare-profiles --codes-file "$OUT/data/train_codes.json" \
  --ontology-file "$OUT/ontology/ontology.jsonl" \
  --umls-terms-file "$OUT/umls/umls_terms.json" --output-dir "$OUT/profiles"
```

`--supplement-dir` is optional and is only used as a fallback for codes absent from the primary ICD-10-CM release. Preparation outputs include provenance and coverage metadata in `manifest.json`, while UMLS-derived terms remain local and are not redistributed.

`prepare-ontology` and MRCONSO `prepare-umls` support ICD-10-CM; ICD-9 Full uses the local-export adapter below. The synthetic pipeline has been verified; real MIMIC end-to-end training and a clean dependency installation have not been verified.

### ICD-9 Full

ICD-9 Full uses MIMIC-III diagnosis and procedure labels. This dataset has **8,692
training-visible labels** (6,724 diagnosis, 1,968 procedure); its full catalog across
splits contains 8,929 labels. Only training labels enter the codebook. Internally,
`DIAG:0031` and `PROC:0031` distinguish diagnosis `003.1` from procedure `00.31`.
Leading zeros are retained. Raw outputs also use these typed keys.

```bash
OUT=artifacts/mimiciii
semicd prepare-dataset --dataset mimiciii --label-space mixed \
  --upstream-file /path/to/mimiciii_full.feather \
  --splits-file /path/to/mimiciii_split.feather --output-dir "$OUT/data"
semicd prepare-icd9-inputs --codes-file "$OUT/data/train_codes.json" \
  --catalog-file /path/to/structured_profiles.csv \
  --paths-file /path/to/official_paths.csv \
  --umls-terms-file /path/to/licensed/icd_synonyms.json \
  --source-contract /path/to/source_contract.json --output-dir "$OUT/inputs"
semicd prepare-profiles --codes-file "$OUT/data/train_codes.json" \
  --ontology-file "$OUT/inputs/ontology.jsonl" \
  --umls-terms-file "$OUT/inputs/umls_terms.json" --output-dir "$OUT/profiles"
```

The adapter accepts existing local exports: catalog CSV columns `code_norm`,
`official_code`, `code_kind` (`diagnosis`/`procedure`), `chapter_text`, `block_text`,
`category_text`, `description`; path CSV columns `code_norm`, `chapter_id`,
`block_id`, `category_id`, `leaf_id`. `code_norm` uses typed keys; `official_code`
and `leaf_id` use matching original dotted codes. The raw UMLS JSON maps original
dotted codes to string lists, **not already selected profile synonyms**.

The source contract may record `ontology_edition` and `umls_release`; missing
values remain unknown in the manifest, with profile edition `unverified`.
The adapter validates supplied catalog/path membership and coverage. It does not
certify an official ontology, infer a source edition, or verify billability;
ICD-9 profiles use catalog membership instead of the ICD-10-CM billability gate.
UMLS exports remain local. Missing terms use the existing description fallback.
Use fresh output directories and inspect `issues.json` and `manifest.json`.
The local 8,692-code export conversion and synthetic CLI/SID round trips have
been checked; real clinical preprocessing and ICD-9 model training have not been rerun.
The prepared dataset manifest must be passed to training with
`--dataset-manifest "$OUT/data/manifest.json"` (also required for real ICD-10-CM training).

## Code layout

| Folder | Steps |
| --- | --- |
| `src/semicd/data/` | 1–2: dataset and code profiles |
| `src/semicd/codebook/` | 3: embeddings, HKM / RK-Means / RQ-VAE, SID bundle |
| `src/semicd/training/` | 4–6: tasks, fine-tuning, constrained decoding, evaluation |


## Acknowledgements

Data preparation follows [Edin et al.](https://github.com/JoakimEdin/medical-coding-reproducibility). The RQ-VAE model is adapted from [LC-Rec](https://github.com/RUCAIBox/LC-Rec), and training uses [Unsloth](https://github.com/unslothai/unsloth). MIMIC, UMLS, and pretrained models remain subject to their own licenses; the code is [MIT-licensed](LICENSE).
