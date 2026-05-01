import torch
import numpy as np

from scOT.problems.base import BaseTimeDataset


class Burgers2D(BaseTimeDataset):
    """Planar exact entropy solutions of scalar Burgers on square 2D grids.

    The field is scalar, but the tensor is a standard scOT image of shape
    [1, H, W]. For direction="x" the exact solution is the 1D entropy solution
    of u_t + (u^2/2)_x = 0 repeated in y. For direction="xy", the planar
    coordinate is (x + y) / sqrt(2), giving a genuine diagonal 2D square field
    while retaining an exact Riemann solution before boundary interaction.
    """

    def __init__(
        self,
        *args,
        resolution=128,
        dt=0.01,
        domain_radius=2.0,
        direction="x",
        case_mix="shock",
        left_min=0.4,
        left_max=1.2,
        right_min=-0.8,
        right_max=0.2,
        interface_min=-0.25,
        interface_max=0.25,
        num_val=256,
        num_test=256,
        n_max=20000,
        seed=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.N_max = int(n_max)
        self.N_val = int(num_val)
        self.N_test = int(num_test)
        self.resolution = int(resolution)
        self.dt = float(dt)
        self.domain_radius = float(domain_radius)
        self.direction = str(direction).lower()
        self.case_mix = str(case_mix).lower()
        self.left_min = float(left_min)
        self.left_max = float(left_max)
        self.right_min = float(right_min)
        self.right_max = float(right_max)
        self.interface_min = float(interface_min)
        self.interface_max = float(interface_max)
        self.seed = int(seed)

        coords = torch.linspace(
            -self.domain_radius,
            self.domain_radius,
            self.resolution,
            dtype=torch.float32,
        )
        self.x, self.y = torch.meshgrid(coords, coords, indexing="ij")
        self.s, self.speed_factor = self.planar_coordinate()

        self.input_dim = 1
        self.label_description = "[u]"
        self.pixel_mask = torch.tensor([False])
        self.constants = {
            "mean": torch.tensor([0.0]).view(1, 1, 1),
            "std": torch.tensor([1.0]).view(1, 1, 1),
            "time": 1.0,
        }
        self.post_init()

    def planar_coordinate(self):
        if self.direction == "x":
            return self.x, 1.0
        if self.direction == "y":
            return self.y, 1.0
        if self.direction in {"xy", "diag", "diagonal"}:
            return (self.x + self.y) / np.sqrt(2.0), np.sqrt(2.0)
        raise ValueError("direction must be x, y, xy, diag, or diagonal.")

    def trajectory_parameters(self, trajectory_index):
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(trajectory_index))
        u_left = self.left_min + (self.left_max - self.left_min) * torch.rand(
            (), generator=generator
        ).item()
        u_right = self.right_min + (self.right_max - self.right_min) * torch.rand(
            (), generator=generator
        ).item()

        if self.case_mix in {"rarefaction", "fan"} and u_left > u_right:
            u_left, u_right = u_right, u_left
        elif self.case_mix in {"mixed", "both"}:
            if torch.rand((), generator=generator).item() < 0.5:
                u_left, u_right = min(u_left, u_right), max(u_left, u_right)
            else:
                u_left, u_right = max(u_left, u_right), min(u_left, u_right)
        else:
            if u_left < u_right:
                u_left, u_right = u_right, u_left

        interface = self.interface_min + (
            self.interface_max - self.interface_min
        ) * torch.rand((), generator=generator).item()
        return float(u_left), float(u_right), float(interface)

    def exact_scalar(self, trajectory_index, time_index):
        u_left, u_right, interface = self.trajectory_parameters(trajectory_index)
        time = float(time_index) * self.dt * self.speed_factor
        s = self.s - interface

        if time <= 0.0:
            return torch.where(s < 0.0, torch.full_like(s, u_left), torch.full_like(s, u_right))

        if u_left > u_right:
            shock_speed = 0.5 * (u_left + u_right)
            return torch.where(
                s < shock_speed * time,
                torch.full_like(s, u_left),
                torch.full_like(s, u_right),
            )

        left_edge = u_left * time
        right_edge = u_right * time
        rarefaction = s / time
        return torch.where(
            s < left_edge,
            torch.full_like(s, u_left),
            torch.where(s > right_edge, torch.full_like(s, u_right), rarefaction),
        )

    def exact_state(self, trajectory_index, time_index):
        return self.exact_scalar(trajectory_index, time_index).unsqueeze(0)

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
