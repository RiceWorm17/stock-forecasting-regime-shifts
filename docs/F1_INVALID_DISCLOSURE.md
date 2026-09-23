# F1/2025 Invalid-Run Disclosure

The original one-time F1/2025 execution remains **`INVALID_STOP_NO_RERUN`**.

Prediction generation and sealing completed, but the protocol's exact-identity verification between the sealed and canonical learned-prediction vectors failed after CSV serialization. A later read-only forensic audit reproduced a deterministic parse-and-reserialize mechanism that changed last-place floating-point representations and reproduced saved overall metrics. That diagnosis did not and cannot convert the run to PASS.

Accordingly:

- F1/2025 does not support any primary or confirmatory scientific claim in this public candidate.
- No F1/2025 performance number or figure appears in the README.
- The exploratory 2025 findings in the claims ledger remain explicitly post-failure and non-confirmatory.
- The original authorization was consumed; no rerun, retraining, inference, or prediction regeneration was performed for this release audit.
- Raw F1 outputs, predictions, metrics, authorization records, signing material, and state records are retained only in the private archive and are not redistributed here.

The unchanged `docs/audit/PHASE_2L_FINAL_FORENSIC_AUDIT_REPORT.md` documents the preserved failure boundary. Its inclusion is for transparency, not evidence of a valid final test.

