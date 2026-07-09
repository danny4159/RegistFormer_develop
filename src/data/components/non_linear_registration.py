"""Non-linear registration methods: ConvexAdam and AnatomIX."""
import sys
import numpy as np
import torch

_CONVEXADAM_SRC = '/SSD2_8TB/Daniel/23_convexadam/convexAdam/src'
if _CONVEXADAM_SRC not in sys.path:
    sys.path.insert(0, _CONVEXADAM_SRC)

from convexAdam.convex_adam_MIND import convex_adam_pt
from convexAdam.apply_convex import apply_convex


def register_convexadam(fixed_np: np.ndarray, moving_np: np.ndarray) -> np.ndarray:
    """
    ConvexAdam non-linear registration using MIND-SSC.

    Args:
        fixed_np: Fixed image (H, W, D) in [-1, 1]
        moving_np: Moving image (H, W, D) in [-1, 1]

    Returns:
        Warped moving image (H, W, D) in [-1, 1]
    """
    fixed_t = torch.from_numpy(fixed_np).float()
    moving_t = torch.from_numpy(moving_np).float()

    disp = convex_adam_pt(
        img_fixed=fixed_t, img_moving=moving_t,
        mind_r=1, mind_d=2, lambda_weight=1.25, grid_sp=6, disp_hw=4,
        selected_niter=80, selected_smooth=0, grid_sp_adam=2,
        ic=True, use_mask=False, dtype=torch.float32, verbose=False,
        device=torch.device('cpu'),
    )
    warped = apply_convex(disp=disp, moving=moving_t.numpy())
    return warped.astype(np.float32)


def register_anatomix(fixed_np: np.ndarray, moving_np: np.ndarray, ckpt_path: str = None) -> np.ndarray:
    """
    AnatomIX non-linear registration (convex-adam with network features).

    Args:
        fixed_np: Fixed image (H, W, D) in [-1, 1]
        moving_np: Moving image (H, W, D) in [-1, 1]
        ckpt_path: Path to anatomix checkpoint (.pth). If None, uses HuggingFace variant.

    Returns:
        Warped moving image (H, W, D) in [-1, 1]
    """
    try:
        from anatomix.registration.run_convex_adam_with_network_feats import convex_adam
        from anatomix.registration.convex_adam_utils import load_model
    except ImportError:
        raise ImportError(
            "AnatomIX not installed. Please install from: "
            "/SSD2_8TB/Daniel/22_anatomix/anatomix"
        )

    import tempfile
    import nibabel as nib
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save images temporarily
        fixed_path = tmpdir / "fixed.nii.gz"
        moving_path = tmpdir / "moving.nii.gz"

        nib.save(nib.Nifti1Image(fixed_np, np.eye(4)), str(fixed_path))
        nib.save(nib.Nifti1Image(moving_np, np.eye(4)), str(moving_path))

        # Run anatomix registration
        convex_adam(
            expname="inline_anatomix",
            lambda_weight=0.75,
            grid_sp=2,
            disp_hw=1,
            selected_niter=80,
            selected_smooth=0,
            ckpt_path=ckpt_path,
            hf_variant='anatomix' if ckpt_path is None else None,
            grid_sp_adam=2,
            ic=True,
            result_path=str(tmpdir),
            fixed_image=str(fixed_path),
            moving_image=str(moving_path),
            use_mask=False,
        )

        # Load warped result
        result_files = list(tmpdir.glob("moved_*.nii.gz"))
        if not result_files:
            raise RuntimeError("AnatomIX did not produce output files")

        result_path = result_files[0]
        warped_np = nib.load(str(result_path)).get_fdata().astype(np.float32)

    return warped_np


def register_nonlinear(
    fixed_np: np.ndarray,
    moving_np: np.ndarray,
    method: str = 'convexadam',
    anatomix_ckpt_path: str = None,
) -> np.ndarray:
    """
    Non-linear registration dispatcher.

    Args:
        fixed_np: Fixed image (H, W, D) in [-1, 1]
        moving_np: Moving image (H, W, D) in [-1, 1]
        method: 'convexadam' (default) or 'anatomix'
        anatomix_ckpt_path: Path to anatomix checkpoint (only for anatomix method)

    Returns:
        Warped moving image (H, W, D) in [-1, 1]
    """
    if method == 'convexadam':
        return register_convexadam(fixed_np, moving_np)
    elif method == 'anatomix':
        return register_anatomix(fixed_np, moving_np, anatomix_ckpt_path)
    else:
        raise ValueError(f"Unknown registration method: {method}")
