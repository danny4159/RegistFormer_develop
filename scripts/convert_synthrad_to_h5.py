"""
Convert SynthRad MR-CT Abdomen NIfTI files to H5 format.

H5 Structure:
- /mr/{patient_name}: MR data
- /ct/{patient_name}: CT data
"""

import os
import h5py
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def convert_split_to_h5(
    input_dir: str,
    output_dir: str,
    split: str,
    mr_filename: str = "mr_preprocess_-1to1.nii.gz",
    ct_filename: str = "ct_preprocess_-1to1.nii.gz",
):
    """
    Convert NIfTI files from a split (train/val/test/try) to H5 format.

    Args:
        input_dir: Base input directory containing split folders
        output_dir: Output directory for H5 files
        split: Split name (train, val, test, try)
        mr_filename: MR NIfTI filename
        ct_filename: CT NIfTI filename
    """
    split_input_dir = Path(input_dir) / split
    if not split_input_dir.exists():
        print(f"Split directory not found: {split_input_dir}")
        return

    # Get all patient folders
    patient_folders = sorted([
        d for d in split_input_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    if len(patient_folders) == 0:
        print(f"No patient folders found in {split_input_dir}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Output H5 file path
    h5_filename = f"{split}set_mr_ct_abdomen.h5"
    h5_path = Path(output_dir) / h5_filename

    print(f"\nConverting {split} split...")
    print(f"Input: {split_input_dir}")
    print(f"Output: {h5_path}")
    print(f"Found {len(patient_folders)} patients")

    with h5py.File(h5_path, 'w') as h5f:
        # Create groups for MR and CT
        mr_group = h5f.create_group('mr')
        ct_group = h5f.create_group('ct')

        for patient_folder in tqdm(patient_folders, desc=f"Processing {split}"):
            patient_name = patient_folder.name

            mr_path = patient_folder / mr_filename
            ct_path = patient_folder / ct_filename

            # Check if both files exist
            if not mr_path.exists():
                print(f"\nWarning: MR file not found for {patient_name}: {mr_path}")
                continue
            if not ct_path.exists():
                print(f"\nWarning: CT file not found for {patient_name}: {ct_path}")
                continue

            # Load NIfTI files
            mr_nii = nib.load(str(mr_path))
            ct_nii = nib.load(str(ct_path))

            mr_data = mr_nii.get_fdata().astype(np.float32)
            ct_data = ct_nii.get_fdata().astype(np.float32)

            # Save to H5
            mr_group.create_dataset(
                patient_name,
                data=mr_data,
                compression='gzip',
                compression_opts=4
            )
            ct_group.create_dataset(
                patient_name,
                data=ct_data,
                compression='gzip',
                compression_opts=4
            )

    print(f"Saved: {h5_path}")

    # Print file size
    file_size_mb = os.path.getsize(h5_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SynthRad MR-CT Abdomen NIfTI files to H5 format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/milab/SSD5_8TB/Daniel/09_registformer_hugging_face/mem_eff_registformer/data/synthrad2025_mr-ct_abdomen",
        help="Input directory containing train/val/test/try folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/milab/SSD5_8TB/Daniel/Daniel_ssd2/RegistFormer/data/SynthRad_MR_CT_Abdomen",
        help="Output directory for H5 files"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs='+',
        default=["train", "val", "test", "try"],
        help="Splits to convert (default: train val test try)"
    )
    parser.add_argument(
        "--mr_filename",
        type=str,
        default="mr_preprocess_-1to1.nii.gz",
        help="MR NIfTI filename"
    )
    parser.add_argument(
        "--ct_filename",
        type=str,
        default="ct_preprocess_-1to1.nii.gz",
        help="CT NIfTI filename"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SynthRad MR-CT Abdomen to H5 Converter")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Splits: {args.splits}")
    print(f"MR filename: {args.mr_filename}")
    print(f"CT filename: {args.ct_filename}")
    print("=" * 60)

    for split in args.splits:
        output_split_dir = os.path.join(args.output_dir, split)
        convert_split_to_h5(
            input_dir=args.input_dir,
            output_dir=output_split_dir,
            split=split,
            mr_filename=args.mr_filename,
            ct_filename=args.ct_filename,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
