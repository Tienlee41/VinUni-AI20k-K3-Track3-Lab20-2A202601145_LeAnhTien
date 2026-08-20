# Benchmark Report

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline | 0.02 | 0.0000 | 4.5 | 100% | 0% | 0 supervisor routes |
| multi-agent | 0.86 | 0.0000 | 8.5 | 100% | 0% | 3 supervisor routes |
| baseline | 0.02 | 0.0000 | 4.5 | 100% | 0% | 0 supervisor routes |
| multi-agent | 0.02 | 0.0000 | 8.5 | 100% | 0% | 3 supervisor routes |
| baseline | 0.02 | 0.0000 | 4.5 | 100% | 0% | 0 supervisor routes |
| multi-agent | 0.02 | 0.0000 | 8.5 | 100% | 0% | 3 supervisor routes |

## Interpretation

Quality is a transparent automated smoke-test score, not a substitute for peer review. Citation coverage counts cited identifiers that match retrieved source IDs.

This checked-in report was generated in offline-corpus mode; therefore its
estimated provider cost is `$0.0000`. A live API run records provider usage in
the same metric fields.

## Failure-mode notes

The main operational risks are retrieval failure, unsupported claims, provider timeouts, and citation mismatch. The workflow records route history, trace events, errors, and token usage so a reviewer can identify which stage needs correction.
