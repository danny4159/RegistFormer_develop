"""Export best Stage 1 predictions for all IXI splits, then train 2D VoxelMorph.

Run this as a detached process with --watch-pid to queue it behind synthesis.
The original HDF5 files are read only. Exported NIfTI and HDF5 arrays use the
same H,W,D axes and float32 [-1,1] scale as the generator, without display
normalization. HDF5 inputs lack scanner geometry, so NIfTI uses voxel axes.
"""

import argparse
import fcntl
import gc
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PROJECT_ROOT", str(ROOT))
LOG = logging.getLogger("ixi_synthesis_to_voxelmorph")
H5_NAME = "IXI_Stage1ForVoxelmorph.h5"
REGISTRATION = {
    "fixed": "synth_T2",
    "moving": "T2_moved_3mm_RigidReg",
    "evaluation_only": "T2",
    "spatial_dims": 2,
    "loss": {"L1": 1.0, "deformation_smoothness": 0.5},
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def wait_for_process(pid, start_ticks, poll_seconds):
    while process_identity(pid) == start_ticks:
        LOG.info("Waiting for synthesis PID %s to finish training and testing", pid)
        time.sleep(poll_seconds)


def select_best_checkpoint(last_checkpoint, required_epoch=None):
    """Use the callback's full-precision validation score, not rounded filenames."""
    import torch

    state = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
    if required_epoch is not None and state["epoch"] < required_epoch:
        raise RuntimeError(
            f"Synthesis stopped at epoch {state['epoch']}; expected >= {required_epoch}"
        )
    candidates = []
    for key, callback in state.get("callbacks", {}).items():
        if "val/ssim_B" not in str(key) or not callback.get("best_model_path"):
            continue
        score = float(callback["best_model_score"])
        if not math.isfinite(score):
            raise RuntimeError("Synthesis best validation SSIM is not finite")
        path = Path(callback["best_model_path"])
        if not path.is_file():
            path = Path(last_checkpoint).parent / path.name
        if not path.is_file():
            raise FileNotFoundError(f"Best synthesis checkpoint is missing: {path}")
        candidates.append((score, path))
    if not candidates:
        raise RuntimeError("No val/ssim_B ModelCheckpoint state was found")
    score, path = max(candidates, key=lambda item: item[0])
    epoch = int(state["epoch"])
    del state
    gc.collect()
    return path, score, epoch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_split(model, input_h5, output_root, split, groups, checkpoint_hash,
                 device, batch_size, max_patients=None, max_slices=None):
    """Round-trip actual NIfTI files into a registration-ready, atomic HDF5."""
    import h5py
    import nibabel as nib
    import numpy as np
    import torch

    output_root = Path(output_root)
    nii_dir = output_root / "nifti" / split
    nii_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_root / split / H5_NAME
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    if h5_path.exists():
        raise FileExistsError(f"Refusing to overwrite an exported split: {h5_path}")
    temporary_h5 = h5_path.with_name(h5_path.name + ".partial")
    source_group, target_group, reference_group = groups
    if len({source_group, target_group, reference_group}) != 3:
        raise ValueError("Expected distinct source, evaluation and reference groups")
    records = []
    with h5py.File(input_h5, "r") as source, h5py.File(temporary_h5, "w") as dest:
        patients = list(source[source_group])
        if max_patients is not None:
            patients = patients[:max_patients]
        for name in groups:
            dest.create_group(name)
            for key, value in source[name].attrs.items():
                dest[name].attrs[key] = value
        dest.create_group("synth_T2")
        for patient in patients:
            if Path(patient).name != patient:
                raise ValueError(f"Unsafe patient identifier: {patient}")
            volumes = {name: source[name][patient][...].astype(np.float32) for name in groups}
            if max_slices is not None:
                volumes = {name: value[:, :, :max_slices] for name, value in volumes.items()}
            shape = volumes[source_group].shape
            if len(shape) != 3 or any(value.shape != shape for value in volumes.values()):
                raise ValueError(f"Input group shapes differ for {split}/{patient}")
            for value in volumes.values():
                if not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite source volume: {split}/{patient}")
            prediction = np.empty(shape, dtype=np.float32)
            with torch.inference_mode():
                for start in range(0, shape[2], batch_size):
                    stop = min(shape[2], start + batch_size)
                    tensors = []
                    for name in (source_group, reference_group):
                        array = np.ascontiguousarray(volumes[name][:, :, start:stop].transpose(2, 0, 1))
                        tensors.append(torch.from_numpy(array).unsqueeze(1).to(device))
                    generated = model(*tensors).detach().cpu().numpy()[:, 0]
                    prediction[:, :, start:stop] = generated.transpose(1, 2, 0)
            if not np.isfinite(prediction).all() or prediction.min() < -1.00001 or prediction.max() > 1.00001:
                raise ValueError(f"Invalid synthesis output: {split}/{patient}")
            nifti_path = nii_dir / f"{patient}_synth_T2.nii.gz"
            temporary_nii = nii_dir / f".{patient}_synth_T2.partial.nii.gz"
            image = nib.Nifti1Image(prediction, np.eye(4))
            image.set_data_dtype(np.float32)
            image.header["descrip"] = b"native HDF5 voxel grid; float32 generator scale [-1,1]"
            nib.save(image, temporary_nii)
            temporary_nii.replace(nifti_path)
            # Read the generated NIfTI, rather than writing the in-memory result
            # directly: HDF5 is explicitly built from the exported image.
            roundtrip = nib.load(nifti_path).get_fdata(dtype=np.float32)
            if not np.array_equal(roundtrip, prediction):
                raise ValueError(f"NIfTI changed the axes or values: {nifti_path}")
            for name, volume in volumes.items():
                dataset = dest[name].create_dataset(patient, data=volume, compression="gzip", compression_opts=1, shuffle=True)
                for key, value in source[name][patient].attrs.items():
                    dataset.attrs[key] = value
            dataset = dest["synth_T2"].create_dataset(patient, data=roundtrip, compression="gzip", compression_opts=1, shuffle=True)
            dataset.attrs["nifti_path"] = str(nifti_path.resolve())
            dataset.attrs["axis_order"] = "HWD"
            records.append({"patient": patient, "shape": list(shape), "nifti": str(nifti_path.resolve())})
            LOG.info("Exported %s/%s: %s", split, patient, shape)
        for key, value in source.attrs.items():
            dest.attrs[key] = value
        dest.attrs["source_h5"] = str(Path(input_h5).resolve())
        dest.attrs["synthesis_checkpoint_sha256"] = checkpoint_hash
        dest.attrs["axis_order"] = "HWD"
        dest.attrs["generated_intensity_scale"] = "float32 [-1,1]; no display normalization"
        dest.attrs["complete"] = True
    temporary_h5.replace(h5_path)
    return {"input_h5": str(Path(input_h5).resolve()), "h5": str(h5_path.resolve()),
            "patients": records, "patient_count": len(records),
            "slice_count": sum(item["shape"][2] for item in records)}


def registration_command(output_dir, run_dir):
    return [sys.executable, str(ROOT / "src/train.py"),
            "experiment=ixi_voxelmorph_2d_synthesis196ed3d",
            f"data.data_dir={Path(output_dir).resolve()}",
            f"hydra.run.dir={Path(run_dir).resolve()}"]


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"pid": os.getpid(), "synthesis_run": str(args.synthesis_run_dir.resolve()),
                  "output_dir": str(output), "registration_run": str(args.registration_run_dir.resolve()),
                  "registration": REGISTRATION}

        def update(phase, **extra):
            status.update(phase=phase, updated_at=datetime.now().isoformat(), **extra)
            write_json(output / "status.json", status)
            LOG.info("Pipeline phase: %s", phase)

        try:
            if args.watch_pid:
                if not args.watch_start_ticks:
                    raise ValueError("--watch-start-ticks is required with --watch-pid")
                update("waiting_for_synthesis", waiting_for_pid=args.watch_pid)
                wait_for_process(args.watch_pid, args.watch_start_ticks, args.poll_seconds)
            import hydra
            import torch
            from omegaconf import OmegaConf

            torch.set_num_threads(2)
            torch.manual_seed(42)
            cfg = OmegaConf.load(args.synthesis_run_dir / ".hydra/config.yaml")
            data = OmegaConf.to_container(cfg.data, resolve=True)
            groups = [data[f"data_group_{i}"] for i in (1, 2, 3)]
            if groups != ["T1", "T2", "T2_moved_3mm_RigidReg"] or data["use_25d_style"]:
                raise ValueError("Unexpected synthesis data routing; refusing a different experiment")
            update("selecting_best_checkpoint")
            if args.checkpoint:
                if not args.export_only:
                    raise ValueError("Explicit --checkpoint is supported only for export-only validation")
                checkpoint, score, epoch = args.checkpoint.resolve(), None, None
            else:
                checkpoint, score, epoch = select_best_checkpoint(
                    args.synthesis_run_dir / "checkpoints/last.ckpt", int(cfg.trainer.max_epochs) - 1)
                console = (args.synthesis_run_dir / "console.log").read_text(errors="replace")
                if "Finished experiment at" not in console:
                    LOG.warning("All synthesis epochs completed but its automatic test did not finish; exporting all splits from the validated best checkpoint")
            provenance = output / "provenance"
            provenance.mkdir(exist_ok=True)
            saved_checkpoint = provenance / "best_synthesis.ckpt"
            if saved_checkpoint.exists():
                raise FileExistsError(f"Refusing to overwrite pipeline provenance: {saved_checkpoint}")
            shutil.copy2(checkpoint, saved_checkpoint)
            checkpoint_hash = sha256(saved_checkpoint)
            shutil.copy2(args.synthesis_run_dir / ".hydra/config.yaml", provenance / "synthesis_config.yaml")
            update("exporting", best_checkpoint=str(checkpoint), best_ssim=score,
                   source_last_epoch=epoch, checkpoint_sha256=checkpoint_hash)
            state = torch.load(saved_checkpoint, map_location="cpu", weights_only=False)
            model = hydra.utils.instantiate(cfg.model.netG_A)
            generator_state = {key[len("netG_A."):]: value for key, value in state["state_dict"].items() if key.startswith("netG_A.")}
            model.load_state_dict(generator_state, strict=True)
            del state, generator_state
            gc.collect()
            model = model.to(args.device).eval()
            manifests = {}
            patient_sets = {}
            for split in ("train", "val", "test"):
                source_h5 = Path(data["data_dir"]) / split / data[f"{split}_file"]
                manifests[split] = export_split(model, source_h5, output, split, groups,
                    checkpoint_hash, args.device, args.inference_batch_size, args.max_patients, args.max_slices)
                patient_sets[split] = {item["patient"] for item in manifests[split]["patients"]}
                update("exporting", completed_splits=list(manifests))
            for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                if patient_sets[left] & patient_sets[right]:
                    raise ValueError(f"Patient leakage between {left} and {right}")
            write_json(output / "manifest.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                "best_validation_ssim": score, "seed": 42, "splits": manifests,
                "registration": REGISTRATION})
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            update("exports_complete")
            if args.export_only:
                return
            if args.max_patients or args.max_slices:
                raise ValueError("Partial exports cannot be used for production registration training")
            registration_dir = args.registration_run_dir.resolve()
            registration_dir.mkdir(parents=True, exist_ok=True)
            if (registration_dir / "launch.json").exists():
                raise FileExistsError("Registration already has a launch record; refusing a duplicate")
            command = registration_command(output, registration_dir)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid,
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", HYDRA_FULL_ERROR="1", PYTHONUNBUFFERED="1")
            write_json(registration_dir / "launch.json", {"argv": command, "dataset": str(output),
                "gpu_uuid": args.gpu_uuid, **REGISTRATION,
                "batch_size": 16, "max_epochs": 100, "inshape": [256, 256], "source_checkpoint": str(checkpoint)})
            shutil.copy2(ROOT / "configs/experiment/ixi_voxelmorph_2d_synthesis196ed3d.yaml", provenance / "registration_experiment.yaml")
            with (registration_dir / "console.log").open("w") as console:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT)
                update("registration_training", registration_pid=child.pid, command=command)
                returncode = child.wait()
            if returncode != 0:
                raise RuntimeError(f"VoxelMorph exited with code {returncode}; see {registration_dir / 'console.log'}")
            update("complete")
        except Exception as error:
            update("failed", error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesis-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registration-run-dir", type=Path, required=True)
    parser.add_argument("--watch-pid", type=int)
    parser.add_argument("--watch-start-ticks")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--gpu-uuid", default="GPU-06e6a8fa-1322-40a0-ff22-bd7897a4d2cf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--max-slices", type=int)
    args = parser.parse_args()
    if args.device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    if args.inference_batch_size < 1 or args.poll_seconds <= 0:
        parser.error("Batch size and polling interval must be positive")
    if any(value is not None and value < 1 for value in (args.max_patients, args.max_slices)):
        parser.error("Patient and slice limits must be positive")
    if (args.max_patients or args.max_slices) and not args.export_only:
        parser.error("Partial exports require --export-only")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
