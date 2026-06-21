"""
Phase 12 — ML confidence layer (SKELETON).

This package adds a MACHINE-LEARNING confidence filter ON TOP of the existing
rule-based SMC SequenceRunner. It does NOT replace the strategy:

    SequenceRunner -> approved A+/A/B setup -> [12.5 ML gate: P_win >= tau] -> alert

The model learns, from historical setups + their realized outcomes, the
conditional probability that an *already-approved* setup reaches TP1 before SL.
It is trained on tabular features (no GPU, no deep learning) — gradient boosting
on the feature vector produced here.

Sub-phases (see the project blueprint):
    12.1  feature_extractor.py  — turn a completed setup into a fixed feature row
    12.2  labeler.py + scripts/build_ml_dataset.py — label setups by outcome,
          fan out across instruments/cores (RunPod-ready) into a training dataset
    12.3+ trainer / evaluator / inference gate — added once 12.2 shows enough data

Nothing here touches the live signal logic. Importing this package has no side
effects and adds no runtime cost to the bot.
"""
