# AMOR-003 — Tokenizer Evaluation

## Baseline

AMOR-B0 uses a byte-aware BPE tokenizer.

Configuration:

- Vocabulary target: 32,000
- Normalization: Unicode NFC
- Pre-tokenizer: ByteLevel
- Decoder: ByteLevel
- Byte fallback: enabled
- Case preservation: enabled
- Special tokens: AMOR control tokens

## Correctness Results

The tokenizer successfully passed:

- Training test
- English input
- Source code
- Mathematical expressions
- JSON/tool commands
- Unicode input
- Encode/decode round-trip

Test result:

7 passed.

## Preliminary Metrics

| Category | Characters | Tokens | Chars/token |
|---|---:|---:|---:|
| English | 26 | 8 | 3.25 |
| Code | 30 | 9 | 3.33 |
| Math | 9 | 5 | 1.80 |
| JSON | 34 | 9 | 3.78 |
| Path | 22 | 9 | 2.44 |
| Unicode | 17 | 7 | 2.43 |

## Interpretation

These metrics are preliminary because the tokenizer was trained
on an extremely small synthetic corpus.

The results are therefore not representative of the final AMOR
tokenizer.

The experiment primarily validates the tokenizer implementation
and evaluation pipeline.

## Important Observation

ByteLevel tokenization exposes internal byte representations when
token IDs are converted directly to token strings. This does not
necessarily indicate Unicode corruption.

Round-trip encode/decode tests passed successfully.

## Next Experiment

Train the AMOR tokenizer on the actual curated training corpus.

Before finalizing the tokenizer vocabulary size, compare:

- 8K
- 16K
- 32K
- 48K

using the same evaluation corpus and metrics.