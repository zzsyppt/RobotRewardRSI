# Phase 8E callable curriculum candidate

This candidate is intentionally minimal. It declares training-side reward,
curriculum, and diagnostic components as public callable contracts without
requesting runner hooks, evaluation wrappers, checkpoint control, or budget
changes.

The intended idea is to let early balance form before ramping perturbation
exposure, while keeping the external runner responsible for all training and
evaluation execution.
