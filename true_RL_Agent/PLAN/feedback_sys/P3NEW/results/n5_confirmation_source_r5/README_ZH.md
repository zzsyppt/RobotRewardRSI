# R5 source frozen; CPU verified; capture not released

The reviewed v2 inventory was snapshotted once: 500 files, 10 changed + 8 added against R4. Local and remote sources match the complete manifest. All remote checks used CUDA_VISIBLE_DEVICES empty. The new reference checks (50) and facts/report checks (8) passed; both actual preflights exited 0. Prior same-SHA report/index evidence is referenced separately, not counted again.

Authoritative refs are in source_freeze_receipt.json: manifest, bundle, downloaded original verification, history, R4 card, commands and root's source/CPU-only authorization. The human R5 card remains DRAFT_NOT_RELEASED. No capture card, actual capture, GPU run, model/API request or ledger was created by this operation; existing STOP files were not cleared.

Stair: n5_r5_20261001; namespace public_dev_diagnostic_n5_r5_20261001; runner seeds 751084905 / 610839053 / 226852600. Push eval_base_seed 1900000000; runner seeds 1900010000 / 1900010001 / 1900010002. Derived seeds are disjoint from recorded R1–R4; cohort hashes differ from the supplied 19 Stair / 21 Push jobs. Both checkpoint SHAs match R4. This is a bounded historical comparison, not an all-server uniqueness claim.

The 33-member original receipt/log/preflight archive was SHA-verified. First local extraction hit a Windows long-path error; its failure record remains. Existing extracted bytes were checked and only missing members were written with extended paths; all 33 now match the archive, without rerunning remote checks or changing source. Future capture requires separate actual resource/budget/STOP review and root release. No semantic qualification is implied.
