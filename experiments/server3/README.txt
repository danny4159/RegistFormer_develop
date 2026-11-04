Server 3 experiment archive
===========================
Recorded: 2026-11-04 (Asia/Seoul)
Host: milab-server3
Workspace: /SSD5_8TB/Daniel/RegistFormer_develop
Base commit: e7ed9a878f7d4584e9723879619e686e093b150b
Archive branch: server3/experiment-cleanup-20261104

This is a preservation/organization commit series of existing working changes.
It does not introduce a new training run or change experimental hyperparameters.
Existing changes have several origins; they are not all AAAI reviewer ablations.

1. Historical 196ed3d pipeline (October 22 experiments)
- configs/experiment/ixi_proposed_synthesis_196ed3d.yaml
- configs/experiment/ixi_voxelmorph_2d_synthesis196ed3d.yaml
- configs/experiment/ixi_registformer_196ed3d.yaml
- Corresponding configs/model/*196ed3d.yaml, src/models/*196ed3d_module.py,
  and src/models/components/network_*196ed3d.py.
- src/ixi_synthesis_to_voxelmorph.py exports best synthesis predictions for all
  splits to NIfTI/H5, then trains 2D VoxelMorph.
- T1 source, T2_moved_3mm_RigidReg reference; aligned T2 evaluation only.
- Registration: synthesized T2 fixed, rigid-registered moved T2 moving;
  L1 + deformation smoothness. This is NOT the later aaai_last K3 model.
- Checkpoint paths in the RegistFormer config are historical SERVER-LOCAL paths;
  weights/data are not in Git. Override those paths on another server.

2. Shared execution and evaluation fixes
- VoxelMorph optional L1 and skip disabled loss terms.
- Registration LPIPS clamps inputs to [-1,1] before metric evaluation.
- Sharpness accepts CPU tensors (its OpenCV computation is already on CPU).
- Consistent lightning.pytorch ModelCheckpoint import; resumed training tests
  the selected checkpoint after fitting rather than automatically the resume input.

3. Transformer experiment option
- pure_use_refine_stage (default true) optionally removes the last full-resolution
  refinement stage; the default architecture is preserved.
- This option alone is not evidence that an ablation was trained/evaluated.
- network_notes/ retains existing CNN and pure-NATTEN flow descriptions. Each
  document identifies its own reference commit; these are descriptive snapshots.

4. AAAI reviewer experiments live in a SEPARATE worktree
Workspace: /SSD5_8TB/Daniel/RegistFormer_aaai_last_ablation
Base: aaai_last, ba1bdedec680dcbb0e41e10be5a71fd7dedafca0
Its uncommitted changes were inspected but are NOT included in this branch.
Its frozen logs/<run>/runtime copies are the source of truth for historical runs,
not the newer MIGS source files in this workspace. See artifact_index.json.
Existing logs, checkpoints, H5/NIfTI and frozen runtime copies remain in place.
The index records their locations; it is not a backup of their contents.

Validation on 2026-11-04
- regist environment, CPU: python -m unittest discover -s tests -v: 8/8 passed.
- These cover export integrity, checkpoint selection, registration input routing,
  L1 gradients, LPIPS range, and deformation-axis/sampling consistency.
- Changed/new Python sources syntax checked; no new GPU training performed.

Local-only material
- .gitignore excludes codes/ (ad-hoc flat source exports), data/logs/checkpoints,
  caches and binaries. Nothing was deleted or reset during this cleanup.
- Branch commits are local; remote publishing is a separate action.
