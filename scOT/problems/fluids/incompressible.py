import torch
import h5py
import numpy as np
import copy
from scOT.problems.base import BaseTimeDataset
from scOT.problems.fluids.normalization_constants import CONSTANTS


class TaylorGreenVortex(BaseTimeDataset):
    """Analytic 2D Taylor-Green vortex for long autoregressive stability tests.

    The default output is the macroscopic velocity field [u, v] on a square
    periodic grid. The exact solution is

        u =  A sin(x + phi_x) cos(y + phi_y) exp(-2 nu t)
        v = -A cos(x + phi_x) sin(y + phi_y) exp(-2 nu t)

    on [0, 2*pi]^2. Density/pressure channels can be included by setting
    just_velocities=False, but the entropy-like regularizers for this benchmark
    should operate on velocity, energy, enstrophy, and divergence.
    """

    def __init__(
        self,
        *args,
        resolution=128,
        dt=0.05,
        viscosity=None,
        reynolds_number=None,
        velocity_scale=1.0,
        length_scale=1.0,
        amplitude_min=0.8,
        amplitude_max=1.2,
        phase_jitter=True,
        tracer=False,
        just_velocities=True,
        num_val=256,
        num_test=256,
        n_max=20000,
        seed=0,
        **kwargs,
    ):
        if tracer:
            raise ValueError("TaylorGreenVortex does not have a tracer")
        super().__init__(*args, **kwargs)
        self.N_max = int(n_max)
        self.N_val = int(num_val)
        self.N_test = int(num_test)
        self.resolution = int(resolution)
        self.dt = float(dt)
        if reynolds_number is not None:
            self.reynolds_number = float(reynolds_number)
            viscosity_from_re = (
                float(velocity_scale) * float(length_scale) / self.reynolds_number
            )
            if viscosity is not None and not np.isclose(float(viscosity), viscosity_from_re):
                raise ValueError(
                    "TaylorGreenVortex received inconsistent viscosity and "
                    "reynolds_number values."
                )
            self.viscosity = viscosity_from_re
        else:
            self.reynolds_number = None
            self.viscosity = 0.01 if viscosity is None else float(viscosity)
        self.amplitude_min = float(amplitude_min)
        self.amplitude_max = float(amplitude_max)
        self.phase_jitter = bool(phase_jitter)
        self.just_velocities = bool(just_velocities)
        self.seed = int(seed)

        coords = torch.arange(self.resolution, dtype=torch.float32)
        coords = coords * (2.0 * np.pi / float(self.resolution))
        self.x, self.y = torch.meshgrid(coords, coords, indexing="ij")

        if self.just_velocities:
            self.input_dim = 2
            self.label_description = "[u,v]"
            self.pixel_mask = torch.tensor([False, False])
            mean = torch.tensor([0.0, 0.0]).view(2, 1, 1)
            std = torch.tensor([0.5, 0.5]).view(2, 1, 1)
        else:
            self.input_dim = 4
            self.label_description = "[rho],[u,v],[p]"
            self.pixel_mask = torch.tensor([False, False, False, False])
            mean = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(4, 1, 1)
            std = torch.tensor([1.0, 0.5, 0.5, 0.25]).view(4, 1, 1)

        # time is already returned in physical units by __getitem__.
        self.constants = {"mean": mean, "std": std, "time": 1.0}
        self.post_init()

    def trajectory_parameters(self, trajectory_index):
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(trajectory_index))
        amplitude = self.amplitude_min + (
            self.amplitude_max - self.amplitude_min
        ) * torch.rand((), generator=generator).item()
        if self.phase_jitter:
            phase_x = 2.0 * np.pi * torch.rand((), generator=generator).item()
            phase_y = 2.0 * np.pi * torch.rand((), generator=generator).item()
        else:
            phase_x = 0.0
            phase_y = 0.0
        return amplitude, phase_x, phase_y

    def exact_state(self, trajectory_index, time_index):
        time = float(time_index) * self.dt
        amplitude, phase_x, phase_y = self.trajectory_parameters(trajectory_index)
        decay = np.exp(-2.0 * self.viscosity * time)
        x = self.x + phase_x
        y = self.y + phase_y
        u = amplitude * torch.sin(x) * torch.cos(y) * decay
        v = -amplitude * torch.cos(x) * torch.sin(y) * decay

        if self.just_velocities:
            return torch.stack([u, v], dim=0)

        rho = torch.ones_like(u)
        pressure = 0.25 * amplitude**2 * (torch.cos(2.0 * x) + torch.cos(2.0 * y))
        pressure = pressure * decay**2
        return torch.stack([rho, u, v, pressure], dim=0)

    def normalize(self, state):
        return (state - self.constants["mean"]) / self.constants["std"]

    def denormalize(self, state):
        return state * self.constants["std"] + self.constants["mean"]

    def __getitem__(self, idx):
        i, t, t1, t2 = self._idx_map(idx)
        trajectory_index = i + self.start
        inputs = self.exact_state(trajectory_index, t1)
        label = self.exact_state(trajectory_index, t2)
        return {
            "pixel_values": self.normalize(inputs),
            "labels": self.normalize(label),
            "time": float(t) * self.dt,
            "pixel_mask": self.pixel_mask,
        }


class IncompressibleBase(BaseTimeDataset):
    def __init__(
        self,
        N_max,
        file_path,
        *args,
        tracer=False,
        just_velocities=False,
        transpose=False,
        resolution=None,
        **kwargs
    ):
        """
        just_velocities: If True, only the velocities are used as input and output.
        transpose: If True, the input and output are transposed.
        """
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.N_max = N_max
        self.N_val = 120
        self.N_test = 240
        self.resolution = 128
        self.tracer = tracer
        self.just_velocities = just_velocities
        self.transpose = transpose

        data_path = self.data_path + file_path
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")

        self.constants = copy.deepcopy(CONSTANTS)
        if just_velocities:
            self.constants["mean"] = self.constants["mean"][1:3]
            self.constants["std"] = self.constants["std"][1:3]

        self.density = torch.ones(1, self.resolution, self.resolution)
        self.pressure = torch.zeros(1, self.resolution, self.resolution)

        self.input_dim = 4 if not tracer else 5
        if just_velocities:
            self.input_dim -= 2
        self.label_description = "[u,v]"
        if not self.just_velocities:
            self.label_description = "[rho],[u,v],[p]"
        if tracer:
            self.label_description += ",[tracer]"

        self.pixel_mask = torch.tensor([False, False])
        if not self.just_velocities:
            self.pixel_mask = torch.tensor([False, False, False, True])
        if tracer:
            self.pixel_mask = torch.cat(
                [self.pixel_mask, torch.tensor([False])],
                dim=0,
            )

        if resolution is None:
            self.res = None
        else:
            if resolution > 128:
                raise ValueError("Resolution must be <= 128")
            self.res = resolution

        self.post_init()

    def _downsample(self, image, target_size):
        image = image.unsqueeze(0)
        image_size = image.shape[-2]
        freqs = torch.fft.fftfreq(image_size, d=1 / image_size)
        sel = torch.logical_and(freqs >= -target_size / 2, freqs <= target_size / 2 - 1)
        image_hat = torch.fft.fft2(image, norm="forward")
        image_hat = image_hat[:, :, sel, :][:, :, :, sel]
        image = torch.fft.ifft2(image_hat, norm="forward").real
        return image.squeeze(0)

    def __getitem__(self, idx):
        i, t, t1, t2 = self._idx_map(idx)
        time = t / self.constants["time"]

        inputs_v = (
            torch.from_numpy(self.reader["velocity"][i + self.start, t1, 0:2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        label_v = (
            torch.from_numpy(self.reader["velocity"][i + self.start, t2, 0:2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        if self.transpose:
            inputs_v = inputs_v.transpose(-2, -1)
            label_v = label_v.transpose(-2, -1)

        if not self.just_velocities:
            inputs = torch.cat([self.density, inputs_v, self.pressure], dim=0)
            label = torch.cat([self.density, label_v, self.pressure], dim=0)
        else:
            inputs = inputs_v
            label = label_v

        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        label = (label - self.constants["mean"]) / self.constants["std"]

        if self.tracer:
            input_tracer = (
                torch.from_numpy(self.reader["velocity"][i + self.start, t1, 2:3])
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )
            output_tracer = (
                torch.from_numpy(self.reader["velocity"][i + self.start, t2, 2:3])
                .type(torch.float32)
                .reshape(1, self.resolution, self.resolution)
            )
            if self.transpose:
                input_tracer = input_tracer.transpose(-2, -1)
                output_tracer = output_tracer.transpose(-2, -1)
            input_tracer = (
                input_tracer - self.constants["tracer_mean"]
            ) / self.constants["tracer_std"]
            output_tracer = (
                output_tracer - self.constants["tracer_mean"]
            ) / self.constants["tracer_std"]

            inputs = torch.cat([inputs, input_tracer], dim=0)
            label = torch.cat([label, output_tracer], dim=0)

        if self.res is not None:
            inputs = self._downsample(inputs, self.res)
            label = self._downsample(label, self.res)

        return {
            "pixel_values": inputs,
            "labels": label,
            "time": time,
            "pixel_mask": self.pixel_mask,
        }


class KolmogorovFlow(BaseTimeDataset):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        assert tracer == False

        self.N_max = 20000
        self.N_val = 120
        self.N_test = 240
        self.resolution = 128
        self.just_velocities = just_velocities

        data_path = self.data_path + "/FNS-KF.nc"
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")

        self.constants = copy.deepcopy(CONSTANTS)
        self.constants["mean"][1] = -2.2424793e-13
        self.constants["mean"][2] = 4.1510376e-12
        self.constants["std"][1] = 0.22017328
        self.constants["std"][2] = 0.22078253
        if just_velocities:
            self.constants["mean"] = self.constants["mean"][1:3]
            self.constants["std"] = self.constants["std"][1:3]

        self.density = torch.ones(1, self.resolution, self.resolution)
        self.pressure = torch.zeros(1, self.resolution, self.resolution)
        X, Y = torch.meshgrid(
            torch.linspace(0, 1, self.resolution),
            torch.linspace(0, 1, self.resolution),
            indexing="ij",
        )
        f = lambda x, y: 0.1 * torch.sin(2.0 * np.pi * (x + y))
        self.forcing = f(X, Y).unsqueeze(0)
        self.constants["mean_forcing"] = -1.2996679288335145e-09
        self.constants["std_forcing"] = 0.0707106739282608
        self.forcing = (self.forcing - self.constants["mean_forcing"]) / self.constants[
            "std_forcing"
        ]

        self.input_dim = 5 if not tracer else 6
        if just_velocities:
            self.input_dim -= 2
        self.label_description = "[u,v],[g]"
        if not self.just_velocities:
            self.label_description = "[rho],[u,v],[p],[g]"
        if tracer:
            self.label_description += ",[tracer]"

        self.pixel_mask = torch.tensor([False, False, False])
        if not self.just_velocities:
            self.pixel_mask = torch.tensor([False, False, False, True, False])
        if tracer:
            self.pixel_mask = torch.cat(
                [self.pixel_mask, torch.tensor([False])],
                dim=0,
            )

        self.post_init()

    def __getitem__(self, idx):
        i, t, t1, t2 = self._idx_map(idx)
        time = t / self.constants["time"]

        inputs_v = (
            torch.from_numpy(self.reader["solution"][i + self.start, t1, 0:2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        label_v = (
            torch.from_numpy(self.reader["solution"][i + self.start, t2, 0:2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )

        if not self.just_velocities:
            inputs = torch.cat([self.density, inputs_v, self.pressure], dim=0)
            label = torch.cat([self.density, label_v, self.pressure], dim=0)
        else:
            inputs = inputs_v
            label = label_v

        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        label = (label - self.constants["mean"]) / self.constants["std"]

        inputs = torch.cat([inputs, self.forcing], dim=0)
        label = torch.cat([label, self.forcing], dim=0)

        return {
            "pixel_values": inputs,
            "labels": label,
            "time": time,
            "pixel_mask": self.pixel_mask,
        }


class BrownianBridge(IncompressibleBase):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        if tracer:
            raise ValueError("BrownianBridge does not have a tracer")
        file_path = "/NS-BB.nc"
        super().__init__(
            20000,
            file_path,
            *args,
            tracer=False,
            just_velocities=just_velocities,
            **kwargs
        )


class PiecewiseConstants(IncompressibleBase):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        file_path = "/NS-PwC.nc"
        super().__init__(
            20000,
            file_path,
            *args,
            tracer=tracer,
            just_velocities=just_velocities,
            **kwargs
        )


class Gaussians(IncompressibleBase):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        if tracer:
            raise ValueError("Gaussians does not have a tracer")
        file_path = "/NS-Gauss.nc"
        super().__init__(
            20000,
            file_path,
            *args,
            tracer=False,
            just_velocities=just_velocities,
            **kwargs
        )


class ShearLayer(IncompressibleBase):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        if tracer:
            raise ValueError("Shear layer does not have a tracer")
        super().__init__(
            40000,
            "/NS-SL.nc",
            *args,
            transpose=True,
            tracer=False,
            just_velocities=just_velocities,
            **kwargs
        )


class VortexSheet(IncompressibleBase):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        if tracer:
            raise ValueError("VortexSheet does not have a tracer")
        file_path = "/NS-SVS.nc"
        super().__init__(
            20000,
            file_path,
            *args,
            tracer=False,
            just_velocities=just_velocities,
            **kwargs
        )


class Sines(IncompressibleBase):
    def __init__(self, *args, tracer=False, just_velocities=False, **kwargs):
        if tracer:
            raise ValueError("Sines does not have a tracer")
        file_path = "/NS-Sines.nc"
        super().__init__(
            20000,
            file_path,
            *args,
            tracer=False,
            just_velocities=just_velocities,
            **kwargs
        )
