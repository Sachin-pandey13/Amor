# AMOR-003 — Tokenizer Strategy

## Baseline Configuration

- **Tokenizer:** BPE
- **Vocabulary:** 32,000 tokens
- **Case:** Preserved
- **Unicode:** Preserved with conservative normalization
- **Byte fallback:** To be investigated and benchmarked
- **Special tokens:** Explicit control tokens
- **Training:** Trained on the AMOR corpus
- **Implementation:** Hugging Face Tokenizers

## Research Questions

1. Is 32K vocabulary optimal for AMOR?
2. How does vocabulary size affect sequence length?
3. How well does the tokenizer represent source code?
4. How well does it represent mathematical notation?
5. How does it handle Unicode and uncommon characters?
6. Does byte fallback improve robustness?
7. What is the effect on training efficiency?

## Candidate Vocabulary Sizes

- 8K
- 16K
- 32K
- 48K

## Initial Decision

AMOR-B0 will use a 32K BPE tokenizer.

The tokenizer will be trained from the AMOR corpus rather than
using a pretrained tokenizer.

## Implementation

The Hugging Face Tokenizers library will provide the optimized
tokenization implementation. AMOR will control the tokenizer
configuration, training corpus, vocabulary, special tokens,
and evaluation.