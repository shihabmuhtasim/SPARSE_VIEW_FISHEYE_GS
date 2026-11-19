# Executed controller snapshot

These are the controller and selector files verified against the completed
Eomer run on 19 November 2026. They preserve the implementation that produced
the published measurements. They contain historical host and storage guards.

Do not launch these files on another account. Use `../code/` and the main README
for the portable entry points. The GPU training engine is reconstructed from
the pinned upstream commit plus `../provenance/training_from_upstream.patch`.
The portable launcher changes paths, validation and reporting orchestration;
it does not change the model, selectors, hyperparameters or test split.
