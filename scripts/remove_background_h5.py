"""
Remove background from [-1, 1] normalized MR data in H5 files using Otsu + feathering.

Method:
  1. Otsu threshold on [0,1]-scaled volume to generate binary foreground mask
  2. Slice-wise binary_fill_holes to close internal cavities
  3. Gaussian feathering on the full volume (background fades to -1, brain blends at boundary)

Output H5 has the same group/dataset structure as the input.
"""

import os
import h5py
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes, gaussian_filter


def remove_background_feather(vol: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    img_01 = (vol + 1) / 2.0
    thresh = threshold_otsu(img_01)
    mask = img_01 > thresh
    mask_filled = np.stack(
        [binary_fill_holes(mask[:, :, i]) for i in range(mask.shape[2])], axis=2
    )
    mask_feather = gaussian_filter(mask_filled.astype(float), sigma=sigma)
    result = vol * mask_feather + (-1.0) * (1 - mask_feather)
    return result.astype(np.float32)


def process_h5(input_path: str, output_path: str, sigma: float = 1.5):
    os.makedirs(Path(output_path).parent, exist_ok=True)

    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        groups = list(f_in.keys())
        print(f"Groups: {groups}")

        for group in groups:
            f_out.create_group(group)
            patients = list(f_in[group].keys())

            for patient in tqdm(patients, desc=group):
                vol = f_in[group][patient][()]
                vol_bg = remove_background_feather(vol, sigma=sigma)
                f_out[group].create_dataset(
                    patient,
                    data=vol_bg,
                    compression='gzip',
                    compression_opts=4,
                )

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Remove background from [-1,1] normalized MR H5 data using Otsu + feathering"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input H5 file path"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output H5 file path"
    )
    parser.add_argument(
        "--sigma", type=float, default=1.5,
        help="Gaussian sigma for feathering (default: 1.5)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MR Background Removal (Otsu + Feathering)")
    print("=" * 60)
    print(f"Input : {args.input}")
    print(f"Output: {args.output}")
    print(f"Sigma : {args.sigma}")
    print("=" * 60)

    process_h5(args.input, args.output, sigma=args.sigma)
    print("Done!")


if __name__ == "__main__":
    main()
