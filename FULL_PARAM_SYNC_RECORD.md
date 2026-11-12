# Full-parameter training source synchronization

Captured: 2026-11-12T02:27:19.660717+00:00

Source launcher: `/mnt/share/y00977881/script/vllm_new/multinode_16p_r3/train.sh`.
Default source: `/mnt/share/y00977881/project/veomni_verl_vllm3.1flash_r3/vllm-ascend-private`.
Target branch: `3.1flash/full_param`. Previous commit: `406107ca98279d81aadaba2057cba53025fc9e3b`.

Match server A5-specific Q/K norm rounding and CaMem pinned host backups, including its existing memcpy behavior. Copy matching server tests.

This commit copies the server implementation; it does not independently fix or
validate its numerical behavior. In particular, CaMem uses pinned CPU backups,
and the server's GBSA metadata uses precision 0. No training or NPU tests run
during synchronization. Server files and running jobs are not modified.

Repository guidance, CI coverage, missing tool links and third-party build
patches are retained. Generated version/build metadata, compiled operators,
caches, logs and backups are not copied. Installed dependencies and compiled
operators are outside this source synchronization. The launcher may override
its default directory through `R3_CODE_ROOT`.

Changed/deleted source and test paths:

- `tests/e2e/pull_request/one_card/test_minimax_qknorm_rope_cache.py`
- `tests/ut/device_allocator/test_camem.py`
- `vllm_ascend/device_allocator/camem.py`
- `vllm_ascend/ops/triton/msa/minimax_qknorm_rope_cache.py`

Retained repository-only differences:

- `.claude/README.md`
- `csrc/cmake/third_party/build/modules/patch/protobuf-hide_absl_symbols.patch`
- `csrc/cmake/third_party/build/modules/patch/protobuf_25.1_change_version.patch`

Validation for the coordinated three-repository synchronization:

- All 3,588 selected production files match server Git-blob hashes; absent
  desktop-only runtime modules were removed. Python AST parsing passed.
- Server snapshot was read again before commit and remained unchanged.
- 74 CPU tests passed: attention 46, native MoE-LoRA 16, parameter refit 10,
  Geometry3k 2. Attention also skipped 19 hardware-dependent cases.
- VeOmni full quality commands passed (773 files formatted). Changed verl
  files passed Ruff. Ascend all-file static hooks passed except formatting
  of a copied test; that test was formatted and checked separately, and the
  changed-file long-function check passed.
- Two copied tests have formatting-only changes; production files retain
  the server bytes. Full checkpoint tests could not collect because the
  local CPU environment lacks Ray. NPU and multi-node tests were not run.
