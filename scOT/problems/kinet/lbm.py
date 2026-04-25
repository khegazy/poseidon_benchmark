import os
import torch
import h5py
import numpy as np
from scOT.problems.base import BaseTimeDataset


class WeaklyCompressibleDataset(BaseTimeDataset):
    """
    Dataset for weakly compressible LBM simulations.

    The HDF5 data contains a single long simulation (B=1) with shape
    (1, C, T, H, W) for each variable. This class windows the time axis
    into non-overlapping pseudo-trajectories compatible with BaseTimeDataset.

    Channels: [density, u, v, vorticity] = 4

    data_path should be the full path to the HDF5 file.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 14

        # data_path is the full path to the HDF5 file
        file_path = self.data_path
        if self.move_to_local_scratch is not None:
            file_name = os.path.basename(file_path)
            dest = os.path.join(self.move_to_local_scratch, file_name)
            RANK = int(os.environ.get("LOCAL_RANK", -1))
            if not os.path.exists(dest) and (RANK == 0 or RANK == -1):
                import shutil
                print(f"Start copying {file_name} to {dest}...")
                shutil.copy(file_path, dest)
                print("Finished data copy.")
            from accelerate.utils import broadcast_object_list
            ls = broadcast_object_list([dest], from_process=0)
            file_path = ls[0]

        self.reader = h5py.File(file_path, "r")

        # Determine total timesteps and window-based trajectory count
        self.T_total = self.reader["density"].shape[2]
        self.window_size = self.max_num_time_steps * self.time_step_size + 1
        self.N_max = self.T_total // self.window_size

        # Split: 75% train / 10% val / 15% test
        self.N_test = max(int(self.N_max * 0.15), 1)
        self.N_val = max(int(self.N_max * 0.10), 1)

        self.resolution = self.reader["density"].shape[3]

        # Compute normalization from training portion
        self.constants = self._compute_normalization()

        # 4 channels: density(1) + velocity(2) + vorticity(1)
        self.input_dim = 4
        self.label_description = "[density],[u,v],[vorticity]"
        self.pixel_mask = torch.tensor([False, False, False, False])

        self.post_init()

    def _compute_normalization(self):
        """Compute mean/std from training windows (first 75%), subsampled."""
        N_train = self.N_max - self.N_val - self.N_test
        T_train = N_train * self.window_size

        # Subsample ~500 timesteps from training portion for efficiency
        n_samples = min(500, T_train)
        indices = np.linspace(0, T_train - 1, n_samples, dtype=int)

        density = self.reader["density"][0, :, indices, :, :]     # (1, n, H, W)
        vel = self.reader["velocity"][0, :, indices, :, :]        # (2, n, H, W)
        vort = self.reader["vorticity"][0, :, indices, :, :]      # (1, n, H, W)

        data = np.concatenate([density, vel, vort], axis=0)  # (4, n, H, W)

        mean = torch.tensor(data.mean(axis=(1, 2, 3)), dtype=torch.float32)
        std = torch.tensor(data.std(axis=(1, 2, 3)), dtype=torch.float32)

        return {
            "mean": mean.unsqueeze(1).unsqueeze(1),  # (4, 1, 1)
            "std": std.unsqueeze(1).unsqueeze(1),      # (4, 1, 1)
            "time": float(self.max_num_time_steps * self.time_step_size),
        }

    def __getitem__(self, idx):
        i, t, t1, t2 = self._idx_map(idx)
        time = t / self.constants["time"]

        # Map window-relative time indices to absolute HDF5 indices
        abs_t1 = (i + self.start) * self.window_size + t1
        abs_t2 = (i + self.start) * self.window_size + t2

        # Read input
        den_in = self.reader["density"][0, :, abs_t1, :, :]      # (1, H, W)
        vel_in = self.reader["velocity"][0, :, abs_t1, :, :]     # (2, H, W)
        vort_in = self.reader["vorticity"][0, :, abs_t1, :, :]   # (1, H, W)
        inputs = (
            torch.from_numpy(np.concatenate([den_in, vel_in, vort_in], axis=0))
            .type(torch.float32)
        )

        # Read label
        den_out = self.reader["density"][0, :, abs_t2, :, :]     # (1, H, W)
        vel_out = self.reader["velocity"][0, :, abs_t2, :, :]    # (2, H, W)
        vort_out = self.reader["vorticity"][0, :, abs_t2, :, :]  # (1, H, W)
        label = (
            torch.from_numpy(np.concatenate([den_out, vel_out, vort_out], axis=0))
            .type(torch.float32)
        )

        # Normalize
        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        label = (label - self.constants["mean"]) / self.constants["std"]

        return {
            "pixel_values": inputs,
            "labels": label,
            "time": time,
            "pixel_mask": self.pixel_mask,
        }


# Alias for get_dataset routing
D2Q9_WeaklyCompressible = WeaklyCompressibleDataset
