"""
ANTsPy-based MR-to-CT registration script.
Registers CT to MR space using SyN algorithm (mutual information metric).
Preserves original MR orientation in the output.
"""
import os
import argparse
import ants
import numpy as np


def normalize_robust(img_ants, pmin=1.0, pmax=99.0):
    """
    Robust intensity normalization to [0, 1] using percentiles.
    Useful for cross-modality registration (MR-CT).
    """
    arr = img_ants.numpy().astype(np.float32)
    lo, hi = np.percentile(arr, [pmin, pmax])
    arr = np.clip(arr, lo, hi)
    if hi - lo < 1e-8:
        arr = arr - lo
    else:
        arr = (arr - lo) / (hi - lo)
    return ants.from_numpy(
        arr, origin=img_ants.origin, spacing=img_ants.spacing, direction=img_ants.direction
    )


def maybe_resample_to_fixed(moving, fixed):
    """Resample moving image to fixed grid if spacing/size differs significantly."""
    return ants.resample_image_to_target(moving, fixed, interp_type="linear")


def register_mr_ct(
    mr_path,
    ct_path,
    output_path,
    syn_type="SyN",
    do_robust_norm=True,
    do_hist_match=False,
):
    """
    Register CT to MR space using ANTsPy.

    Args:
        mr_path: Path to fixed MR image (nii.gz)
        ct_path: Path to moving CT image (nii.gz)
        output_path: Path to save warped CT image (ct_syn.nii.gz)
        syn_type: Type of registration ("SyN", "SyNCC", "Affine", etc.)
        do_robust_norm: Apply robust intensity normalization (recommended for cross-modality)
        do_hist_match: Apply histogram matching (optional, not always beneficial for CT-MR)

    Returns:
        Tuple of (output_path, registration_dict)
    """
    print("\n" + "=" * 70)
    print("MR-CT Registration with ANTsPy")
    print("=" * 70)

    # Load images
    print("\n[Loading images]")
    fixed = ants.image_read(mr_path)   # MR (fixed/reference)
    moving = ants.image_read(ct_path)  # CT (moving)

    print(f"Fixed (MR) shape: {fixed.shape}, spacing: {fixed.spacing}")
    print(f"Moving (CT) shape: {moving.shape}, spacing: {moving.spacing}")

    # Store original MR orientation for later restoration
    fixed_direction = fixed.direction
    fixed_spacing = fixed.spacing
    fixed_origin = fixed.origin

    # Resample moving to fixed grid if needed
    print("\n[Resampling moving image to fixed grid]")
    moving_rs = maybe_resample_to_fixed(moving, fixed)

    # Robust normalization for cross-modality
    print("\n[Normalizing images]")
    if do_robust_norm:
        fixed_p = normalize_robust(fixed)
        moving_p = normalize_robust(moving_rs)
        print("Robust normalization applied (percentile-based)")
    else:
        fixed_p, moving_p = fixed, moving_rs

    # Optional histogram matching
    if do_hist_match:
        print("[Histogram matching]")
        moving_p = ants.histogram_match_image(moving_p, fixed_p)

    # Registration
    print(f"\n[Running {syn_type} registration]")
    print("Using mutual information metric (default for SyN with cross-modality data)")
    reg = ants.registration(
        fixed=fixed_p,
        moving=moving_p,
        type_of_transform=syn_type,
        verbose=True
    )

    # Apply transforms to original moving image (preserves original intensity range)
    print("\n[Applying transforms to original CT image]")
    warped = ants.apply_transforms(
        fixed=fixed,  # Output grid/space based on fixed MR
        moving=moving,  # Original CT image
        transformlist=reg["fwdtransforms"],
        interpolator="linear"
    )

    # Ensure output has same direction/orientation as fixed MR
    print("\n[Preserving MR orientation in output]")
    warped.set_direction(fixed_direction)

    # Save warped CT
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ants.image_write(warped, output_path)

    print("\n" + "=" * 70)
    print("[Registration completed successfully]")
    print("=" * 70)
    print(f"Output image: {output_path}")
    print(f"Output shape: {warped.shape}")
    print(f"Output spacing: {warped.spacing}")
    print(f"Output direction preserved from MR: {np.allclose(warped.direction, fixed_direction)}")
    print(f"\nForward transforms (CT -> MR): {reg['fwdtransforms']}")
    print(f"Inverse transforms (MR -> CT): {reg['invtransforms']}")

    return output_path, reg


def main():
    parser = argparse.ArgumentParser(
        description="Register CT to MR space using ANTsPy SyN algorithm"
    )
    parser.add_argument(
        "--mr",
        type=str,
        default="/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis/one_patient/1PC085/mr_preprocess_-1to1.nii.gz",
        help="Path to fixed MR image"
    )
    parser.add_argument(
        "--ct",
        type=str,
        default="/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis/one_patient/1PC085/ct_preprocess_-1to1.nii.gz",
        help="Path to moving CT image"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2023_mr-ct_pelvis/one_patient/1PC085/ct_syn.nii.gz",
        help="Path to save warped CT image"
    )
    parser.add_argument(
        "--syn_type",
        type=str,
        default="SyN",
        help='Type of transform: "SyN" (default), "SyNCC", "SyNOnly", "Affine", etc.'
    )
    parser.add_argument(
        "--no_robust_norm",
        action="store_true",
        help="Disable robust intensity normalization"
    )
    parser.add_argument(
        "--hist_match",
        action="store_true",
        help="Enable histogram matching"
    )

    args = parser.parse_args()

    register_mr_ct(
        mr_path=args.mr,
        ct_path=args.ct,
        output_path=args.output,
        syn_type=args.syn_type,
        do_robust_norm=not args.no_robust_norm,
        do_hist_match=args.hist_match,
    )


if __name__ == "__main__":
    main()
