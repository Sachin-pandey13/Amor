# AMOR-006 — Training Corpus Plan

## 1. Objective

Build a high-quality, legally usable corpus for training AMOR-B0.

The corpus should support:

- General language understanding
- Reasoning
- Mathematics
- Programming
- Science
- Instruction following
- Structured data
- Tool-oriented interactions

---

## 2. Corpus Principles

AMOR will prioritize:

1. Data quality over raw volume
2. Clear licensing and provenance
3. Deduplication
4. Removal of low-quality content
5. Balanced domain coverage
6. Reproducible preprocessing
7. Separate training and validation data

---

## 3. Initial Corpus Categories

### General text

Purpose:

- Natural language understanding
- Vocabulary
- General knowledge

### Programming

Purpose:

- Code understanding
- Code generation
- Programming reasoning

### Mathematics

Purpose:

- Mathematical language
- Symbolic reasoning
- Problem solving

### Science

Purpose:

- Scientific terminology
- Explanations
- Technical reasoning

### Instruction data

Purpose:

- Following user instructions
- Question answering
- Structured responses

### Tool-oriented data

Purpose:

- Learning structured actions
- Function/tool calling patterns
- JSON generation

---

## 4. Data Quality Pipeline

Raw data:

    ↓

Source validation

    ↓

Format normalization

    ↓

Language filtering

    ↓

Quality filtering

    ↓

PII/sensitive-data filtering

    ↓

Exact deduplication

    ↓

Near-duplicate filtering

    ↓

Domain classification

    ↓

Train/validation split

    ↓

Tokenizer training

    ↓

Tokenization

    ↓

Final training dataset

---

## 5. Data Provenance

Every source should record:

- Dataset/source name
- Source URL
- License
- Download date
- Version
- Processing version
- Number of documents
- Number of tokens

No dataset should enter the final corpus without known provenance.

---

## 6. Git Policy

The following should NOT be committed to Git:

- Large raw datasets
- Personal data
- User interaction data
- Tokenized training shards
- Model checkpoints
- Model weights

Git should contain:

- Dataset acquisition scripts
- Processing code
- Dataset metadata
- Configuration
- Documentation
- Small synthetic samples

---

## 7. Train/Validation Split

Target:

- 98% training
- 2% validation

The split must happen before tokenization where practical.

Validation data must not leak into training.

---

## 8. Initial Training Target

AMOR-B0 has approximately:

12.92M parameters.

The first real training run will intentionally be a relatively small experiment.

We will establish:

- Corpus size
- Token count
- Training steps
- Batch size
- Effective batch size
- Learning rate
- Validation loss

before scaling up.

---

## 9. Reproducibility

Every corpus build should record:

- Random seed
- Source versions
- Filtering configuration
- Deduplication configuration
- Tokenizer version
- Processing code version

The same inputs and configuration should produce the same corpus.

---

## 10. Current Status

AMOR-B0 architecture:

COMPLETE

Training pipeline:

COMPLETE

Tiny-batch overfit:

PASSED

Real corpus:

NEXT