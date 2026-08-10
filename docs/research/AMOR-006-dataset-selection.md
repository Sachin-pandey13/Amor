# AMOR-006 — Dataset Selection

## 1. Objective

Select the initial open datasets for training AMOR-B0.

AMOR-B0 currently has approximately 12.92M trainable parameters.

The first real training experiment will intentionally use a limited corpus so that we can validate:

* Data quality
* Training stability
* Validation loss
* Domain coverage
* Tokenizer performance
* GPU training performance

The initial corpus is a controlled experiment, not the final AMOR dataset.

---

## 2. Initial Corpus Target

Initial target:

**~55–60M tokens**

This is a target rather than a guaranteed final size.

The final number of tokens will depend on:

* Dataset availability
* Filtering
* Deduplication
* Language filtering
* License/provenance checks
* Document quality
* Tokenizer statistics

We will record the actual final token count after preprocessing.

---

## 3. Selected Sources

### 3.1 General English — FineWeb

Dataset:

`HuggingFaceFW/fineweb`

Purpose:

* General language modeling
* Vocabulary coverage
* General knowledge
* Natural English text

Initial target:

**~30M tokens**

License:

**ODC-By 1.0**

FineWeb contains cleaned and deduplicated English web data derived from Common Crawl. The dataset card currently reports more than 18.5T tokens and identifies the dataset license as ODC-By 1.0.

Important:

The dataset-level license does not mean that every underlying web document is free of third-party rights. AMOR must retain dataset provenance and apply appropriate filtering before training.

Source:

https://huggingface.co/datasets/HuggingFaceFW/fineweb

---

### 3.2 Educational Text — FineWeb-Edu

Dataset:

`HuggingFaceFW/fineweb-edu`

Purpose:

* Educational language
* Explanatory writing
* Academic vocabulary
* Higher-quality instructional text

Initial target:

**~10M tokens**

License:

**ODC-By 1.0**

Important:

FineWeb-Edu is distributed under ODC-By 1.0, while the individual document content remains subject to the rights of the original publishers.

Therefore AMOR will retain provenance and will not treat the dataset as equivalent to a public-domain corpus.

Source:

https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

---

### 3.3 Programming — Common Pile Stack v2

Dataset:

`common-pile/stackv2`

Preferred filtered variant:

`common-pile/stackv2_edu_filtered`

Purpose:

* Programming syntax
* Code understanding
* Software engineering vocabulary
* Technical text

Initial target:

**~10M tokens**

The filtered Stack V2 variant is designed to include code from openly licensed repositories and provides per-document license information in metadata.

The dataset creators also explicitly warn that licensing metadata can contain errors, so AMOR must preserve the original license metadata and avoid assuming that every document is automatically safe for unrestricted reuse.

Source:

https://huggingface.co/datasets/common-pile/stackv2_edu_filtered

---

### 3.4 Mathematics — FineMath

Dataset:

`HuggingFaceTB/finemath`

Purpose:

* Mathematical language
* Mathematical reasoning
* Equations
* Problem-solving text
* STEM vocabulary

Initial target:

**~5M tokens**

License:

The FineMath datasets are listed with ODC-By licensing on Hugging Face.

The exact subset/version used by AMOR must be recorded in the corpus manifest before acquisition.

Source:

https://huggingface.co/HuggingFaceTB/finemath

---

### 3.5 Instruction Data — Aya Dataset

Dataset:

`CohereLabs/aya_dataset`

Purpose:

* Instruction following
* Question answering
* Multilingual examples
* Structured prompt/response patterns

Initial target:

**~2–5M tokens**

License:

**Apache 2.0**

Aya Dataset contains more than 204K human-annotated prompt/completion pairs across many languages.

Important:

Instruction data will remain a relatively small component of AMOR's initial corpus.

The first AMOR model is primarily a base language model. Instruction following will eventually be handled more heavily during a separate instruction-tuning stage.

Source:

https://huggingface.co/datasets/CohereLabs/aya_dataset

---

## 4. Initial Domain Mixture

Target distribution:

| Domain          | Target tokens | Approx. share |
| --------------- | ------------: | ------------: |
| General English |           30M |          ~50% |
| Educational     |           10M |          ~17% |
| Programming     |           10M |          ~17% |
| Mathematics     |            5M |           ~8% |
| Instructions    |          2–5M |         ~3–8% |
| **Total**       |   **~55–60M** |      **100%** |

These percentages are initial targets.

The actual distribution will be calculated after preprocessing.

---

## 5. Why General Text Dominates

AMOR-B0 is initially being trained as a base language model.

Therefore the corpus should primarily contain natural language rather than instruction/response pairs.

The intended progression is:

```text
Pretraining
    ↓
General language + knowledge + code + math
    ↓
Base AMOR model
    ↓
Instruction tuning
    ↓
Instruction-following AMOR
```

Instruction datasets should therefore not dominate the pretraining corpus.

---

## 6. Acquisition Strategy

AMOR will not download the complete source datasets.

Large datasets such as FineWeb are far larger than required for the first experiment.

Instead:

```text
Dataset
   ↓
Streaming / controlled sampling
   ↓
Select required documents
   ↓
Save raw selected records
   ↓
Normalize
   ↓
Filter
   ↓
Deduplicate
   ↓
Split
   ↓
Tokenize
   ↓
Training shards
```

The acquisition code must record the source dataset and version used for every corpus build.

---

## 7. Provenance Requirements

Every acquired document should retain, where available:

* Dataset name
* Dataset configuration/subset
* Dataset version or revision
* Original document ID
* Original URL
* Source
* License
* Acquisition timestamp
* Processing version

Example metadata:

```json
{
  "dataset": "HuggingFaceFW/fineweb",
  "subset": "sample-10BT",
  "source_id": "...",
  "url": "...",
  "license": "ODC-By-1.0",
  "acquired_at": "...",
  "processing_version": "AMOR-006"
}
```

---

## 8. Data Processing

All selected data must pass through the AMOR data pipeline.

Processing stages:

```text
Raw records
    ↓
Schema normalization
    ↓
Text extraction
    ↓
Empty/invalid document filtering
    ↓
Language filtering
    ↓
Quality filtering
    ↓
Sensitive/PII filtering
    ↓
Exact deduplication
    ↓
Near-duplicate filtering
    ↓
Domain tagging
    ↓
Train/validation split
```

The existing data pipeline will be extended rather than replaced.

---

## 9. Deduplication

The corpus must use both:

### Exact deduplication

Remove documents with identical normalized text.

### Near-duplicate detection

Detect highly similar documents that differ only through:

* Formatting
* Minor edits
* Boilerplate
* Repeated crawled copies

Deduplication must happen before the final tokenization stage.

---

## 10. Train/Validation Split

Initial target:

```text
Training:   98%
Validation:  2%
```

The validation set must be isolated from training.

Validation data must never be used for:

* Model training
* Token-level optimization
* Hyperparameter selection after repeated inspection

The split should be deterministic using a recorded random seed.

---

## 11. Git Storage Policy

The following must NOT be committed to Git:

* Full raw datasets
* Large downloaded corpora
* Personal data
* User interaction data
* Tokenized training shards
* Model checkpoints
* Model weights

Git should contain:

* Acquisition code
* Processing code
* Dataset manifests
* Dataset metadata
* Configuration
* Research documentation
* Small synthetic samples
* Tests

---

## 12. Corpus Manifest

Every corpus build should produce a manifest containing:

```text
Corpus ID
Build timestamp
Git commit
Dataset sources
Dataset revisions
Licenses
Document counts
Filtered document counts
Duplicate counts
Training documents
Validation documents
Training tokens
Validation tokens
Tokenizer version
Processing configuration
Random seed
```

Example:

```json
{
  "corpus_id": "AMOR-006-v1",
  "git_commit": "...",
  "tokenizer": "AMOR-tokenizer-v1",
  "train_tokens": 0,
  "validation_tokens": 0,
  "seed": 42
}
```

The zero values will be replaced after the corpus is actually built.

---

## 13. Initial Training Constraint

AMOR-B0 has approximately 12.92M parameters.

The first real corpus experiment is intentionally limited to approximately 55–60M tokens.

We will first evaluate:

* Training loss
* Validation loss
* Perplexity
* Loss convergence
* Domain performance
* GPU utilization
* Tokens/second
* Peak GPU memory

Only after this baseline is understood will the corpus or training duration be scaled.

---

## 14. Current Status

AMOR-B0 architecture:

**COMPLETE**

Tokenizer:

**COMPLETE**

Data pipeline:

**COMPLETE**

Training pipeline:

**COMPLETE**

Tiny-batch overfit:

**PASSED**

Corpus strategy:

**COMPLETE**

Dataset selection:

**AMOR-006B — CURRENT**

Next:

**Implement controlled dataset acquisition and provenance tracking.**
