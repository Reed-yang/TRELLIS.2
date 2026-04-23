"""coart.eval — val-set deep evaluation utilities.

Provides:
    - metrics: CD/NC/F-score/topology wrappers around scripts/eval/
    - deep_eval: run_deep_eval() entry point
    - watchdog: alert conditions

These modules dynamically add `scripts/eval/` to sys.path so the canonical
metric implementations can be reused without copying.
"""
