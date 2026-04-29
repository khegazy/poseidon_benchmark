"""Macroscopic decoded-field regularizers for autoregressive PDE rollouts.

These losses operate only on physical fields produced by the model decoder.
"""

import importlib
from collections.abc import Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import ConcatDataset


def _plain_dict(config):
    if config is None:
        return None
    if hasattr(config, "items"):
        return {k: _plain_dict(v) for k, v in dict(config).items()}
    if isinstance(config, list):
        return [_plain_dict(v) for v in config]
    return config


def _first_dataset(dataset):
    if isinstance(dataset, ConcatDataset):
        return _first_dataset(dataset.datasets[0])
    return dataset


def _dataset_time_scale(dataset) -> float:
    dataset = _first_dataset(dataset)
    constants = getattr(dataset, "constants", None)
    if constants is None:
        return 1.0
    return float(constants.get("time", 1.0))


def _dataset_normalization(dataset):
    dataset = _first_dataset(dataset)
    constants = getattr(dataset, "constants", None)
    if constants is None:
        return None, None
    return constants.get("mean", None), constants.get("std", None)


def _normalize_name(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


_DEFAULT_CHANNEL_MAP = {
    "rho": [0],
    "density": [0],
    "h": [0],
    "height": [0],
    "velocity": [1, 2],
    "uv": [1, 2],
    "u": [1],
    "v": [2],
    "momentum": [1, 2],
    "mx": [1],
    "my": [2],
    "hu": [1],
    "hv": [2],
    "energy": [3],
    "totalenergy": [3],
    "p": [3],
    "pressure": [3],
    "omega": [3],
    "vorticity": [3],
}


def _dataset_channel_map(dataset):
    dataset = _first_dataset(dataset)
    descriptors = getattr(dataset, "printable_channel_description", None)
    channel_slices = getattr(dataset, "channel_slice_list", None)
    if descriptors is None or channel_slices is None:
        return {}

    mapping = {}
    for i, descriptor in enumerate(descriptors):
        channels = list(range(channel_slices[i], channel_slices[i + 1]))
        key = _normalize_name(descriptor)
        mapping[key] = channels
        if key in {"density", "rho"}:
            mapping.setdefault("rho", channels)
            mapping.setdefault("density", channels)
        elif key in {"uv", "velocity"} and len(channels) >= 2:
            mapping.setdefault("velocity", channels)
            mapping.setdefault("uv", channels)
            mapping.setdefault("u", [channels[0]])
            mapping.setdefault("v", [channels[1]])
        elif key in {"pressure", "p"}:
            mapping.setdefault("p", channels)
            mapping.setdefault("pressure", channels)
        elif key in {"vorticity", "omega", "w"}:
            mapping.setdefault("omega", channels)
            mapping.setdefault("vorticity", channels)
    return mapping


def _as_channel_buffer(value, name):
    if value is None:
        return None

    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.view(1, 1, 1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.view(1, -1, 1, 1)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [], [C], [C,H,W], or [1,C,H,W].")
    return tensor


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def gradient_periodic_2d(field, dx: float, dy: float):
    grad_x = (
        torch.roll(field, shifts=-1, dims=-1) - torch.roll(field, shifts=1, dims=-1)
    ) / (2.0 * dx)
    grad_y = (
        torch.roll(field, shifts=-1, dims=-2) - torch.roll(field, shifts=1, dims=-2)
    ) / (2.0 * dy)
    return grad_x, grad_y


def divergence_periodic_2d(qx, qy, dx: float, dy: float):
    dqx_dx, _ = gradient_periodic_2d(qx, dx, dy)
    _, dqy_dy = gradient_periodic_2d(qy, dx, dy)
    return dqx_dx + dqy_dy


class PhysicsRegularizer(nn.Module):
    def __init__(self, config=None, dataset=None):
        super().__init__()
        self.config = _plain_dict(config) or {}
        self.dataset = _first_dataset(dataset)
        self.channel_map = _dataset_channel_map(self.dataset)
        self.last_components = {}

        mean, std = _dataset_normalization(self.dataset)
        self.register_buffer("mean", _as_channel_buffer(mean, "mean"), persistent=False)
        self.register_buffer("std", _as_channel_buffer(std, "std"), persistent=False)

    def cfg(self, key, default=None):
        return self.config.get(key, default)

    def cfg_any(self, keys, default=None):
        for key in keys:
            if key in self.config:
                return self.config[key]
        return default

    def weight(self, *keys, default=0.0):
        return float(self.cfg_any(keys + ("weight", "lambda"), default))

    def denormalize(self, state):
        channels = state.shape[1]
        if self.mean is None or self.std is None:
            return state

        mean = self.mean.to(device=state.device, dtype=state.dtype)
        std = self.std.to(device=state.device, dtype=state.dtype)
        if mean.shape[1] != 1:
            mean = mean[:, :channels]
        if std.shape[1] != 1:
            std = std[:, :channels]
        return state * std + mean

    def pressure_offset(self):
        if "pressure_offset" in self.config:
            return float(self.config["pressure_offset"])
        if "mean_pressure" in self.config:
            return float(self.config["mean_pressure"])
        return float(getattr(self.dataset, "mean_pressure", 0.0) or 0.0)

    def resolve_channels(self, value, default):
        value = default if value is None else value
        values = _as_list(value)
        channels = []
        for item in values:
            if isinstance(item, str):
                key = _normalize_name(item)
                if key in self.channel_map:
                    channels.extend(self.channel_map[key])
                elif key in _DEFAULT_CHANNEL_MAP:
                    channels.extend(_DEFAULT_CHANNEL_MAP[key])
                else:
                    raise ValueError(f"Unknown channel descriptor '{item}'.")
            else:
                channels.append(int(item))
        return channels

    def scalar(self, state, spec, default):
        channels = self.resolve_channels(spec, default)
        if len(channels) != 1:
            raise ValueError(f"Expected one scalar channel, got {channels}.")
        return state[:, channels[0]]

    def vector(self, state, spec, default, length):
        channels = self.resolve_channels(spec, default)
        if len(channels) != length:
            raise ValueError(f"Expected {length} channels, got {channels}.")
        return state[:, channels]

    def positive_part(self, value, tau=None):
        tau = float(self.cfg("tau", 1e-3) if tau is None else tau)
        if tau <= 0.0:
            return F.relu(value)
        return tau * F.softplus(value / tau)

    def reshape_dt(self, dt, reference):
        if not torch.is_tensor(dt):
            dt = torch.as_tensor(dt, device=reference.device, dtype=reference.dtype)
        else:
            dt = dt.to(device=reference.device, dtype=reference.dtype)
        while dt.ndim < reference.ndim:
            dt = dt.unsqueeze(-1)
        return torch.clamp(dt * float(self.cfg("dt_scale", 1.0)), min=1e-12)

    def normalize_field(self, value, scale):
        if not bool(self.cfg("normalize", True)):
            return value
        while scale.ndim < value.ndim:
            scale = scale.unsqueeze(-1)
        return value / torch.clamp(scale, min=1e-8)

    def normalize_integral_loss(self, loss, before, after):
        if not bool(self.cfg("normalize", True)):
            return loss
        scale = 0.5 * (before.detach().abs() + after.detach().abs()) + 1e-8
        return loss / scale.pow(2)


class CompositePhysicsRegularizer(nn.Module):
    def __init__(self, regularizers):
        super().__init__()
        self.regularizers = nn.ModuleList([r for r in regularizers if r is not None])
        self.last_components = {}

    def forward(self, previous_state, next_state, dt, labels=None, step_index=None):
        total = next_state.new_zeros(())
        components = {}
        for regularizer in self.regularizers:
            value = regularizer(
                previous_state,
                next_state,
                dt,
                labels=labels,
                step_index=step_index,
            )
            total = total + value
            for key, component in getattr(regularizer, "last_components", {}).items():
                components[key] = component
        self.last_components = components
        return total

    def rollout_loss(self, states, labels=None):
        total = states[-1].new_zeros(())
        components = {}
        for regularizer in self.regularizers:
            if not hasattr(regularizer, "rollout_loss"):
                continue
            value = regularizer.rollout_loss(states, labels=labels)
            total = total + value
            for key, component in getattr(regularizer, "last_components", {}).items():
                components[key] = component
        self.last_components = components
        return total


class DensityPositivity2D(PhysicsRegularizer):
    def forward(self, previous_state, next_state, dt, labels=None, step_index=None):
        weight = self.weight("lambda_density", "lambda_pos")
        if weight <= 0.0:
            return next_state.new_zeros(())
        state = self.denormalize(next_state)
        rho = self.scalar(state, self.cfg("density_channel", "rho"), [0])
        floor = float(self.cfg("density_floor", 1e-6))
        loss = self.positive_part(floor - rho).pow(2).mean()
        self.last_components = {"density_positivity": loss.detach()}
        return weight * loss


class EntropyFluxRegularizer2D(PhysicsRegularizer):
    """Base for convex entropy-entropy flux inequalities on decoded fields."""

    default_component_prefix = "entropy_flux"

    def component_prefix(self):
        return str(self.cfg("component_prefix", self.default_component_prefix))

    def entropy_and_flux(self, state):
        raise NotImplementedError

    def entropy_residual_correction(self, state, eta):
        return None

    def entropy_components(self, state):
        return None

    def relative_entropy(self, predicted_state, target_state):
        return None

    def entropy_positive_part(self, value):
        tau = self.cfg("entropy_tau", self.cfg("inequality_tau", 0.0))
        return self.positive_part(value, tau=tau)

    def entropy_flux(self, previous_state, next_state):
        flux_state = 0.5 * (previous_state + next_state)
        return self.entropy_and_flux(flux_state)

    def entropy_correction_state(self, previous_state, next_state):
        return 0.5 * (previous_state + next_state)

    def residual(self, previous_state, next_state, dt):
        eta_prev, _, _ = self.entropy_and_flux(previous_state)
        eta_next, _, _ = self.entropy_and_flux(next_state)
        eta_flux, qx, qy = self.entropy_flux(previous_state, next_state)

        dt = self.reshape_dt(dt, eta_next)
        deta_dt = (eta_next - eta_prev) / dt
        div_q = divergence_periodic_2d(
            qx,
            qy,
            dx=float(self.cfg("dx", 1.0)),
            dy=float(self.cfg("dy", 1.0)),
        )
        residual = deta_dt + div_q
        correction_state = self.entropy_correction_state(previous_state, next_state)
        correction = self.entropy_residual_correction(correction_state, eta_flux)
        if correction is not None:
            residual = residual + correction

        scale = deta_dt.detach().abs().flatten(1).mean(dim=1)
        scale = scale + div_q.detach().abs().flatten(1).mean(dim=1)
        if correction is not None:
            scale = scale + correction.detach().abs().flatten(1).mean(dim=1)
        return self.normalize_field(residual, scale)

    def local_loss(self, previous_state, next_state, dt):
        residual = self.residual(previous_state, next_state, dt)
        return self.entropy_positive_part(residual).pow(2).mean()

    def global_loss(self, previous_state, next_state):
        entropy_prev = self.entropy_integral(previous_state)
        entropy_next = self.entropy_integral(next_state)
        loss = self.entropy_positive_part(entropy_next - entropy_prev).pow(2)
        loss = self.normalize_integral_loss(loss, entropy_prev, entropy_next)
        return loss.mean()

    def entropy_integral(self, state):
        eta, _, _ = self.entropy_and_flux(state)
        cell_area = float(self.cfg("dx", 1.0)) * float(self.cfg("dy", 1.0))
        return eta.sum(dim=(-2, -1)) * cell_area

    def entropy_component_integrals(self, state):
        components = self.entropy_components(state)
        if not components:
            return None
        cell_area = float(self.cfg("dx", 1.0)) * float(self.cfg("dy", 1.0))
        return {
            name: value.sum(dim=(-2, -1)) * cell_area
            for name, value in components.items()
        }

    def _entropy_integral_scale(self, entropies, entropy_ref):
        normalization = str(self.cfg("rollout_normalization", "initial")).lower()
        if normalization == "max":
            return entropies.detach().abs().max(dim=0).values
        if normalization in {"initial", "start"}:
            return entropy_ref.detach().abs()
        raise ValueError("rollout_normalization must be initial or max.")

    def _entropy_tolerance(self, scale, *keys):
        rel = float(self.cfg_any(keys + ("entropy_tolerance",), 0.0))
        absolute = float(
            self.cfg_any(
                tuple(f"{key}_abs" for key in keys) + ("entropy_tolerance_abs",),
                0.0,
            )
        )
        return absolute + rel * scale

    def _final_label_state(self, labels, reference_state):
        if labels is None:
            return None
        if labels.ndim == reference_state.ndim + 1:
            labels = labels[:, -1]
        if labels.ndim != reference_state.ndim:
            raise ValueError(
                "Entropy rollout budget labels must have shape [B, C, H, W] "
                "or [B, T, C, H, W]."
            )
        return labels.to(device=reference_state.device, dtype=reference_state.dtype)

    def _rollout_ceiling_loss(self, entropies):
        mode = str(self.cfg("rollout_mode", "endpoint")).lower()
        entropy_ref = entropies[0]
        scale = self._entropy_integral_scale(entropies, entropy_ref)

        if mode == "per_step":
            losses = []
            for entropy_prev, entropy_next in zip(entropies[:-1], entropies[1:]):
                step_scale = 0.5 * (
                    entropy_prev.detach().abs() + entropy_next.detach().abs()
                )
                tolerance = self._entropy_tolerance(step_scale, "rollout_tolerance")
                loss = self.entropy_positive_part(
                    entropy_next - entropy_prev - tolerance
                ).pow(2)
                loss = self.normalize_integral_loss(loss, entropy_prev, entropy_next)
                losses.append(loss)
            return torch.stack(losses, dim=0).mean()

        tolerance = self._entropy_tolerance(scale, "rollout_tolerance")
        if mode == "endpoint":
            violation = self.entropy_positive_part(entropies[-1] - entropy_ref - tolerance)
        elif mode in {"max", "running_max"}:
            violation = self.entropy_positive_part(entropies.max(dim=0).values - entropy_ref - tolerance)
        elif mode in {"cumulative", "ladder"}:
            step_scale = 0.5 * (
                entropies[:-1].detach().abs() + entropies[1:].detach().abs()
            )
            step_tolerance = self._entropy_tolerance(step_scale, "rollout_tolerance")
            increases = self.entropy_positive_part(
                entropies[1:] - entropies[:-1] - step_tolerance
            )
            violation = increases.sum(dim=0)
        else:
            raise ValueError(
                "rollout_mode must be endpoint, max/running_max, cumulative/ladder, or per_step."
            )

        loss = violation.pow(2)
        if bool(self.cfg("normalize", True)):
            loss = loss / torch.clamp(scale, min=1e-8).pow(2)
        return loss.mean()

    def _target_budget_loss(self, entropies, labels, reference_state):
        label_state = self._final_label_state(labels, reference_state)
        if label_state is None:
            return None

        entropy_initial = entropies[0]
        entropy_pred_final = entropies[-1]
        entropy_true_final = self.entropy_integral(label_state)

        pred_delta = entropy_pred_final - entropy_initial
        true_delta = (entropy_true_final - entropy_initial.detach()).detach()
        mismatch = pred_delta - true_delta

        loss = mismatch.pow(2)
        if bool(self.cfg("normalize", True)):
            normalization = str(
                self.cfg(
                    "budget_normalization",
                    self.cfg("target_budget_normalization", "initial"),
                )
            ).lower()
            if normalization in {"initial", "start"}:
                scale = entropy_initial.detach().abs()
            elif normalization in {"target", "truth", "label"}:
                scale = entropy_true_final.detach().abs()
            elif normalization in {"max", "all"}:
                scale = torch.maximum(
                    torch.maximum(
                        entropy_initial.detach().abs(),
                        entropy_pred_final.detach().abs(),
                    ),
                    entropy_true_final.detach().abs(),
                )
            elif normalization in {"change", "target_change", "delta"}:
                scale = true_delta.detach().abs()
            else:
                raise ValueError(
                    "budget_normalization must be initial, target, max, or change."
                )

            tolerance = self._entropy_tolerance(
                torch.clamp(scale, min=1e-8),
                "budget_tolerance",
                "target_budget_tolerance",
            )
            if torch.any(tolerance > 0):
                loss = self.positive_part(mismatch.abs() - tolerance, tau=0.0).pow(2)
            loss = loss / torch.clamp(scale, min=1e-8).pow(2)
        return loss.mean()

    def _selected_component_names(self, components):
        requested = self.cfg(
            "component_budget_components",
            self.cfg("entropy_component_budget_components", None),
        )
        if requested is None:
            return list(components.keys())
        names = _as_list(requested)
        unknown = [name for name in names if name not in components]
        if unknown:
            raise ValueError(
                f"Unknown entropy component(s) {unknown}; available components are "
                f"{list(components.keys())}."
            )
        return names

    def _component_weight(self, name):
        weights = self.cfg(
            "component_budget_weights",
            self.cfg("entropy_component_budget_weights", {}),
        )
        if isinstance(weights, Mapping):
            return float(weights.get(name, 1.0))
        return 1.0

    def _target_component_budget_loss(
        self,
        states,
        labels,
        entropies,
        reference_state,
    ):
        label_state = self._final_label_state(labels, reference_state)
        if label_state is None:
            return None, {}

        initial_components = self.entropy_component_integrals(states[0])
        pred_components = self.entropy_component_integrals(states[-1])
        true_components = self.entropy_component_integrals(label_state)
        if not initial_components or not pred_components or not true_components:
            return None, {}

        entropy_ref = entropies[0]
        total_scale = self._entropy_integral_scale(entropies, entropy_ref)
        normalization = str(
            self.cfg(
                "component_budget_normalization",
                self.cfg("entropy_component_budget_normalization", "total"),
            )
        ).lower()

        losses = []
        components = {}
        for name in self._selected_component_names(initial_components):
            weight = self._component_weight(name)
            if weight <= 0.0:
                continue

            initial = initial_components[name]
            pred_delta = pred_components[name] - initial
            true_delta = (true_components[name] - initial.detach()).detach()
            mismatch = pred_delta - true_delta

            if normalization in {"total", "entropy"}:
                scale = total_scale
            elif normalization in {"component", "own"}:
                scale = torch.maximum(
                    torch.maximum(
                        initial.detach().abs(),
                        pred_components[name].detach().abs(),
                    ),
                    true_components[name].detach().abs(),
                )
            elif normalization in {"change", "target_change", "delta"}:
                scale = true_delta.detach().abs()
            else:
                raise ValueError(
                    "component_budget_normalization must be total, component, or change."
                )

            loss = mismatch.pow(2)
            if bool(self.cfg("normalize", True)):
                tolerance = self._entropy_tolerance(
                    torch.clamp(scale, min=1e-8),
                    "component_budget_tolerance",
                    "entropy_component_budget_tolerance",
                )
                if torch.any(tolerance > 0):
                    loss = self.positive_part(
                        mismatch.abs() - tolerance,
                        tau=0.0,
                    ).pow(2)
                loss = loss / torch.clamp(scale, min=1e-8).pow(2)
            loss = loss.mean()
            losses.append(weight * loss)
            components[f"{name}_budget"] = loss.detach()

        if not losses:
            return None, components
        return torch.stack(losses).mean(), components

    def relative_entropy_loss(self, next_state, labels):
        target_state = self._final_label_state(labels, next_state)
        if target_state is None:
            return None
        relative = self.relative_entropy(next_state, target_state)
        if relative is None:
            return None
        loss = relative.mean()
        if bool(self.cfg("relative_entropy_normalize", self.cfg("normalize", True))):
            eta_target, _, _ = self.entropy_and_flux(target_state)
            scale = eta_target.detach().abs().flatten(1).mean(dim=1)
            loss = self.normalize_field(relative, scale).mean()
        return loss

    def rollout_loss(self, states, labels=None):
        rollout_weight = self.weight(
            "lambda_rollout",
            "lambda_cumulative",
            "lambda_entropy_rollout",
            "lambda_entropy_cumulative",
            default=0.0,
        )
        budget_weight = self.weight(
            "lambda_budget",
            "lambda_entropy_budget",
            "lambda_rollout_budget",
            "lambda_target_budget",
            "lambda_entropy_target_budget",
            default=0.0,
        )
        component_budget_weight = self.weight(
            "lambda_component_budget",
            "lambda_entropy_component_budget",
            default=0.0,
        )
        if (
            rollout_weight <= 0.0
            and budget_weight <= 0.0
            and component_budget_weight <= 0.0
        ) or len(states) < 2:
            self.last_components = {}
            return states[-1].new_zeros(())

        entropies = torch.stack([self.entropy_integral(state) for state in states], dim=0)

        prefix = self.component_prefix()
        total = states[-1].new_zeros(())
        components = {}

        if rollout_weight > 0.0:
            rollout = self._rollout_ceiling_loss(entropies)
            total = total + rollout_weight * rollout
            components[f"{prefix}_rollout"] = rollout.detach()

        if budget_weight > 0.0:
            budget = self._target_budget_loss(entropies, labels, states[-1])
            if budget is not None:
                total = total + budget_weight * budget
                components[f"{prefix}_budget"] = budget.detach()

        if component_budget_weight > 0.0:
            component_budget, component_values = self._target_component_budget_loss(
                states,
                labels,
                entropies,
                states[-1],
            )
            for name, value in component_values.items():
                components[f"{prefix}_{name}"] = value
            if component_budget is not None:
                total = total + component_budget_weight * component_budget
                components[f"{prefix}_component_budget"] = component_budget.detach()

        self.last_components = components
        return total

    def forward(self, previous_state, next_state, dt, labels=None, step_index=None):
        total = next_state.new_zeros(())
        components = {}
        prefix = self.component_prefix()

        local_weight = self.weight("lambda_local", "lambda_entropy_local")
        if local_weight > 0.0:
            local = self.local_loss(previous_state, next_state, dt)
            total = total + local_weight * local
            components[f"{prefix}_local"] = local.detach()

        global_weight = self.weight("lambda_global", "lambda_entropy_global")
        if global_weight > 0.0:
            global_loss = self.global_loss(previous_state, next_state)
            total = total + global_weight * global_loss
            components[f"{prefix}_global"] = global_loss.detach()

        relative_weight = self.weight(
            "lambda_relative",
            "lambda_relative_entropy",
            "lambda_entropy_relative",
        )
        if relative_weight > 0.0:
            relative = self.relative_entropy_loss(next_state, labels)
            if relative is not None:
                total = total + relative_weight * relative
                components[f"{prefix}_relative"] = relative.detach()

        self.last_components = components
        return total


class IsothermalEulerEntropyFlux2D(EntropyFluxRegularizer2D):
    """Entropy-entropy flux inequality for barotropic/isothermal Euler fields.

    For decoded macroscopic fields rho, u, v and p = a rho, this computes
    eta = 0.5 rho |u|^2 + a rho log(rho)
    q = u (eta + p)
    and penalizes only positive violations of d_t eta + div q <= 0.
    """

    def pressure_coefficient(self):
        return float(
            self.cfg_any(
                ("pressure_coefficient", "eos_pressure_coefficient", "sound_speed_squared", "cs2"),
                1.0,
            )
        )

    def split_state(self, state):
        state = self.denormalize(state)
        rho = self.scalar(state, self.cfg("density_channel", "rho"), [0])
        velocity = self.vector(state, self.cfg("velocity_channels", "velocity"), [1, 2], 2)
        return rho, velocity[:, 0], velocity[:, 1]

    def entropy_and_flux(self, state):
        rho, u, v = self.split_state(state)
        rho_safe = torch.clamp(rho, min=float(self.cfg("density_floor", 1e-6)))
        cs2 = self.pressure_coefficient()
        kinetic = 0.5 * rho_safe * (u.pow(2) + v.pow(2))
        pressure_potential = cs2 * rho_safe * torch.log(rho_safe)
        eta = kinetic + pressure_potential
        pressure = cs2 * rho_safe
        return eta, u * (eta + pressure), v * (eta + pressure)

    def entropy_components(self, state):
        rho, u, v = self.split_state(state)
        rho_safe = torch.clamp(rho, min=float(self.cfg("density_floor", 1e-6)))
        cs2 = self.pressure_coefficient()
        return {
            "kinetic_energy": 0.5 * rho_safe * (u.pow(2) + v.pow(2)),
            "pressure_potential": cs2 * rho_safe * torch.log(rho_safe),
        }

    def relative_entropy(self, predicted_state, target_state):
        rho_pred, u_pred, v_pred = self.split_state(predicted_state)
        rho_target, u_target, v_target = self.split_state(target_state)
        floor = float(self.cfg("density_floor", 1e-6))
        rho_pred = torch.clamp(rho_pred, min=floor)
        rho_target = torch.clamp(rho_target, min=floor)
        cs2 = self.pressure_coefficient()

        velocity_part = 0.5 * rho_pred * (
            (u_pred - u_target).pow(2) + (v_pred - v_target).pow(2)
        )
        density_part = cs2 * (
            rho_pred * torch.log(rho_pred / rho_target) - rho_pred + rho_target
        )
        return velocity_part + density_part

    def log_mean(self, left, right):
        floor = float(self.cfg("density_floor", 1e-6))
        left = torch.clamp(left, min=floor)
        right = torch.clamp(right, min=floor)
        diff = right - left
        log_diff = torch.log(right) - torch.log(left)
        arithmetic = 0.5 * (left + right)
        close = diff.abs() <= 1e-6 * torch.maximum(left.abs(), right.abs())
        denominator = torch.where(
            log_diff.abs() < 1e-12,
            torch.ones_like(log_diff),
            log_diff,
        )
        return torch.where(close, arithmetic, diff / denominator)

    def entropy_flux(self, previous_state, next_state):
        average = str(self.cfg("entropy_flux_average", "log_mean")).lower()
        if average not in {"log_mean", "logarithmic", "tadmor"}:
            return super().entropy_flux(previous_state, next_state)

        rho_prev, u_prev, v_prev = self.split_state(previous_state)
        rho_next, u_next, v_next = self.split_state(next_state)
        rho = self.log_mean(rho_prev, rho_next)
        u = 0.5 * (u_prev + u_next)
        v = 0.5 * (v_prev + v_next)
        cs2 = self.pressure_coefficient()
        eta = 0.5 * rho * (u.pow(2) + v.pow(2)) + cs2 * rho * torch.log(rho)
        pressure = cs2 * rho
        return eta, u * (eta + pressure), v * (eta + pressure)


IsothermalEulerEntropy2D = IsothermalEulerEntropyFlux2D


class ScalarBurgersEntropyFlux2D(EntropyFluxRegularizer2D):
    """Quadratic entropy pair for scalar Burgers embedded on a 2D grid."""

    default_component_prefix = "burgers_entropy_flux"

    def entropy_and_flux(self, state):
        state = self.denormalize(state)
        u = self.scalar(state, self.cfg("scalar_channel", self.cfg("u_channel", "u")), [0])
        eta = 0.5 * u.pow(2)
        q = u.pow(3) / 3.0
        direction = str(self.cfg("flux_direction", "x")).lower()
        if direction == "x":
            return eta, q, torch.zeros_like(q)
        if direction == "y":
            return eta, torch.zeros_like(q), q
        if direction in {"xy", "both"}:
            return eta, q, q
        raise ValueError("flux_direction must be x, y, xy, or both.")


class ShallowWaterEntropyFlux2D(EntropyFluxRegularizer2D):
    """Total-energy entropy pair for 2D shallow water."""

    default_component_prefix = "shallow_water_entropy_flux"

    def primitive(self, state):
        state = self.denormalize(state)
        height = self.scalar(state, self.cfg("height_channel", "h"), [0])
        momentum = self.vector(state, self.cfg("momentum_channels", "momentum"), [1, 2], 2)
        h_safe = torch.clamp(height, min=float(self.cfg("height_floor", 1e-6)))
        u = momentum[:, 0] / h_safe
        v = momentum[:, 1] / h_safe
        return h_safe, u, v

    def entropy_and_flux(self, state):
        h, u, v = self.primitive(state)
        gravity = float(self.cfg("gravity", 1.0))
        eta = 0.5 * h * (u.pow(2) + v.pow(2)) + 0.5 * gravity * h.pow(2)
        pressure_energy = 0.5 * gravity * h.pow(2)
        return eta, u * (eta + pressure_energy), v * (eta + pressure_energy)


class CompressibleEulerEntropyFlux2D(EntropyFluxRegularizer2D):
    """Ideal-gas mathematical entropy pair for compressible Euler fields."""

    default_component_prefix = "compressible_euler_entropy_flux"

    def primitive(self, state):
        state = self.denormalize(state)
        gamma = float(self.cfg("gamma", 1.4))
        layout = str(self.cfg("state_layout", self.cfg("layout", "conserved"))).lower()

        rho = self.scalar(state, self.cfg("density_channel", "rho"), [0])
        rho_safe = torch.clamp(rho, min=float(self.cfg("density_floor", 1e-6)))
        if layout in {"primitive", "rho_uv_p", "rho-u-v-p"}:
            velocity = self.vector(state, self.cfg("velocity_channels", "velocity"), [1, 2], 2)
            pressure = self.scalar(state, self.cfg("pressure_channel", "pressure"), [3])
            pressure = pressure + self.pressure_offset()
            return rho_safe, velocity[:, 0], velocity[:, 1], torch.clamp(
                pressure,
                min=float(self.cfg("pressure_floor", 1e-8)),
            )

        momentum = self.vector(state, self.cfg("momentum_channels", "momentum"), [1, 2], 2)
        total_energy = self.scalar(state, self.cfg("energy_channel", "energy"), [3])
        u = momentum[:, 0] / rho_safe
        v = momentum[:, 1] / rho_safe
        kinetic = 0.5 * rho_safe * (u.pow(2) + v.pow(2))
        pressure = (gamma - 1.0) * (total_energy - kinetic)
        pressure = torch.clamp(pressure, min=float(self.cfg("pressure_floor", 1e-8)))
        return rho_safe, u, v, pressure

    def mathematical_entropy(self, rho, pressure):
        gamma = float(self.cfg("gamma", 1.4))
        specific_entropy = torch.log(pressure) - gamma * torch.log(rho)
        return -self.entropy_scale() * rho * specific_entropy

    def entropy_scale(self):
        gamma = float(self.cfg("gamma", 1.4))
        return float(self.cfg("entropy_scale", 1.0 / max(gamma - 1.0, 1e-12)))

    def entropy_and_flux(self, state):
        rho, u, v, pressure = self.primitive(state)
        eta = self.mathematical_entropy(rho, pressure)
        return eta, eta * u, eta * v


class CompressibleNavierStokesEntropyFlux2D(CompressibleEulerEntropyFlux2D):
    """Compressible Navier-Stokes entropy flux with viscous/thermal production."""

    default_component_prefix = "compressible_ns_entropy_flux"

    def temperature(self, rho, pressure):
        gas_constant = float(self.cfg("gas_constant", 1.0))
        return torch.clamp(
            pressure / (gas_constant * rho),
            min=float(self.cfg("temperature_floor", 1e-8)),
        )

    def entropy_and_flux(self, state):
        rho, u, v, pressure = self.primitive(state)
        eta = self.mathematical_entropy(rho, pressure)
        qx = eta * u
        qy = eta * v
        entropy_scale = self.entropy_scale()

        kappa = float(self.cfg("thermal_conductivity", self.cfg("kappa", 0.0)))
        if kappa > 0.0:
            temperature = self.temperature(rho, pressure)
            dTdx, dTdy = gradient_periodic_2d(
                temperature,
                dx=float(self.cfg("dx", 1.0)),
                dy=float(self.cfg("dy", 1.0)),
            )
            qx = qx + entropy_scale * kappa * dTdx / temperature
            qy = qy + entropy_scale * kappa * dTdy / temperature
        return eta, qx, qy

    def entropy_residual_correction(self, state, eta):
        rho, u, v, pressure = self.primitive(state)
        temperature = self.temperature(rho, pressure)
        mu = float(self.cfg("viscosity", self.cfg("dynamic_viscosity", 0.0)))
        kappa = float(self.cfg("thermal_conductivity", self.cfg("kappa", 0.0)))
        if mu <= 0.0 and kappa <= 0.0:
            return None

        correction = torch.zeros_like(eta)
        if mu > 0.0:
            du_dx, du_dy = gradient_periodic_2d(
                u,
                dx=float(self.cfg("dx", 1.0)),
                dy=float(self.cfg("dy", 1.0)),
            )
            dv_dx, dv_dy = gradient_periodic_2d(
                v,
                dx=float(self.cfg("dx", 1.0)),
                dy=float(self.cfg("dy", 1.0)),
            )
            div_u = du_dx + dv_dy
            bulk = float(self.cfg("bulk_viscosity", -2.0 * mu / 3.0))
            tau11 = 2.0 * mu * du_dx + bulk * div_u
            tau22 = 2.0 * mu * dv_dy + bulk * div_u
            tau12 = mu * (du_dy + dv_dx)
            viscous_production = (
                tau11 * du_dx + tau22 * dv_dy + tau12 * (du_dy + dv_dx)
            ) / temperature
            correction = correction + viscous_production

        if kappa > 0.0:
            dTdx, dTdy = gradient_periodic_2d(
                temperature,
                dx=float(self.cfg("dx", 1.0)),
                dy=float(self.cfg("dy", 1.0)),
            )
            correction = correction + kappa * (
                dTdx.pow(2) + dTdy.pow(2)
            ) / temperature.pow(2)

        if bool(self.cfg("clamp_entropy_production", True)):
            correction = F.relu(correction)
        return (
            correction
            * self.entropy_scale()
            * float(self.cfg("entropy_production_scale", 1.0))
        )


class CompressiblePositivity2D(PhysicsRegularizer):
    """Macroscopic positivity constraints for compressible primitive quantities."""

    def raw_density_pressure_temperature(self, state):
        state = self.denormalize(state)
        gamma = float(self.cfg("gamma", 1.4))
        gas_constant = float(self.cfg("gas_constant", 1.0))
        layout = str(self.cfg("state_layout", self.cfg("layout", "conserved"))).lower()

        rho = self.scalar(state, self.cfg("density_channel", "rho"), [0])
        rho_safe = torch.clamp(rho, min=float(self.cfg("density_floor", 1e-6)))
        if layout in {"primitive", "rho_uv_p", "rho-u-v-p"}:
            pressure = self.scalar(state, self.cfg("pressure_channel", "pressure"), [3])
            pressure = pressure + self.pressure_offset()
        else:
            momentum = self.vector(state, self.cfg("momentum_channels", "momentum"), [1, 2], 2)
            total_energy = self.scalar(state, self.cfg("energy_channel", "energy"), [3])
            u = momentum[:, 0] / rho_safe
            v = momentum[:, 1] / rho_safe
            kinetic = 0.5 * rho_safe * (u.pow(2) + v.pow(2))
            pressure = (gamma - 1.0) * (total_energy - kinetic)
        temperature = pressure / (gas_constant * rho_safe)
        return rho, pressure, temperature

    def forward(self, previous_state, next_state, dt, labels=None, step_index=None):
        base_weight = self.weight("lambda_pos", "lambda_positivity", default=0.0)
        density_weight = self.weight("lambda_density", "lambda_density_pos", default=base_weight)
        pressure_weight = self.weight("lambda_pressure", "lambda_pressure_pos", default=base_weight)
        temperature_weight = self.weight(
            "lambda_temperature",
            "lambda_temperature_pos",
            default=base_weight,
        )
        if density_weight <= 0.0 and pressure_weight <= 0.0 and temperature_weight <= 0.0:
            return next_state.new_zeros(())

        rho, pressure, temperature = self.raw_density_pressure_temperature(next_state)
        total = next_state.new_zeros(())
        components = {}
        if density_weight > 0.0:
            floor = float(self.cfg("density_floor", 1e-6))
            loss = self.positive_part(floor - rho).pow(2).mean()
            total = total + density_weight * loss
            components["density_positivity"] = loss.detach()
        if pressure_weight > 0.0:
            floor = float(self.cfg("pressure_floor", 1e-8))
            loss = self.positive_part(floor - pressure).pow(2).mean()
            total = total + pressure_weight * loss
            components["pressure_positivity"] = loss.detach()
        if temperature_weight > 0.0:
            floor = float(self.cfg("temperature_floor", 1e-8))
            loss = self.positive_part(floor - temperature).pow(2).mean()
            total = total + temperature_weight * loss
            components["temperature_positivity"] = loss.detach()
        self.last_components = components
        return total


class CustomEntropyFlux2D(EntropyFluxRegularizer2D):
    """Adapter for user-supplied macroscopic entropy pair callables.

    The callable is configured as "module.submodule:name" and must return
    (eta, qx, qy). By default it receives denormalized decoded fields and this
    regularizer instance.
    """

    default_component_prefix = "custom_entropy_flux"

    def __init__(self, config=None, dataset=None):
        super().__init__(config=config, dataset=dataset)
        path = self.cfg("entropy_pair", self.cfg("callable", None))
        if path is None:
            raise ValueError("custom_entropy_flux_2d requires entropy_pair.")
        module_name, name = str(path).split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        self.entropy_pair = getattr(module, name)

    def entropy_and_flux(self, state):
        if not bool(self.cfg("custom_receives_normalized", False)):
            state = self.denormalize(state)
        result = self.entropy_pair(state, self)
        if len(result) != 3:
            raise ValueError("Custom entropy_pair must return (eta, qx, qy).")
        return result


class ContinuityResidual2D(PhysicsRegularizer):
    def residual(self, previous_state, next_state, dt):
        previous = self.denormalize(previous_state)
        next_ = self.denormalize(next_state)
        rho_prev = self.scalar(previous, self.cfg("density_channel", "rho"), [0])
        rho_next = self.scalar(next_, self.cfg("density_channel", "rho"), [0])
        vel_prev = self.vector(previous, self.cfg("velocity_channels", "velocity"), [1, 2], 2)
        vel_next = self.vector(next_, self.cfg("velocity_channels", "velocity"), [1, 2], 2)

        rho_mid = 0.5 * (rho_prev + rho_next)
        vel_mid = 0.5 * (vel_prev + vel_next)
        dt = self.reshape_dt(dt, rho_next)
        drho_dt = (rho_next - rho_prev) / dt
        div_flux = divergence_periodic_2d(
            rho_mid * vel_mid[:, 0],
            rho_mid * vel_mid[:, 1],
            dx=float(self.cfg("dx", 1.0)),
            dy=float(self.cfg("dy", 1.0)),
        )
        residual = drho_dt + div_flux
        scale = drho_dt.detach().abs().flatten(1).mean(dim=1)
        scale = scale + div_flux.detach().abs().flatten(1).mean(dim=1)
        residual = self.normalize_field(residual, scale)
        return residual

    def forward(self, previous_state, next_state, dt, labels=None, step_index=None):
        weight = self.weight("lambda_continuity", "lambda_mass")
        if weight <= 0.0:
            return next_state.new_zeros(())
        residual = self.residual(previous_state, next_state, dt)
        loss = residual.pow(2).mean()
        self.last_components = {"continuity": loss.detach()}
        return weight * loss


class IsothermalEulerMomentumResidual2D(PhysicsRegularizer):
    def pressure_coefficient(self):
        return float(
            self.cfg_any(
                ("pressure_coefficient", "eos_pressure_coefficient", "sound_speed_squared", "cs2"),
                1.0,
            )
        )

    def residuals(self, previous_state, next_state, dt):
        previous = self.denormalize(previous_state)
        next_ = self.denormalize(next_state)
        rho_prev = self.scalar(previous, self.cfg("density_channel", "rho"), [0])
        rho_next = self.scalar(next_, self.cfg("density_channel", "rho"), [0])
        vel_prev = self.vector(previous, self.cfg("velocity_channels", "velocity"), [1, 2], 2)
        vel_next = self.vector(next_, self.cfg("velocity_channels", "velocity"), [1, 2], 2)

        rho_mid = 0.5 * (rho_prev + rho_next)
        vel_mid = 0.5 * (vel_prev + vel_next)
        rho_safe = torch.clamp(rho_mid, min=float(self.cfg("density_floor", 1e-6)))
        cs2 = self.pressure_coefficient()
        pressure = cs2 * rho_safe
        u = vel_mid[:, 0]
        v = vel_mid[:, 1]

        mx_prev = rho_prev * vel_prev[:, 0]
        mx_next = rho_next * vel_next[:, 0]
        my_prev = rho_prev * vel_prev[:, 1]
        my_next = rho_next * vel_next[:, 1]

        dt = self.reshape_dt(dt, mx_next)
        dmx_dt = (mx_next - mx_prev) / dt
        div_mx = divergence_periodic_2d(
            rho_safe * u * u + pressure,
            rho_safe * u * v,
            dx=float(self.cfg("dx", 1.0)),
            dy=float(self.cfg("dy", 1.0)),
        )
        dmy_dt = (my_next - my_prev) / dt
        div_my = divergence_periodic_2d(
            rho_safe * u * v,
            rho_safe * v * v + pressure,
            dx=float(self.cfg("dx", 1.0)),
            dy=float(self.cfg("dy", 1.0)),
        )
        rx = dmx_dt + div_mx
        ry = dmy_dt + div_my

        scale_x = dmx_dt.detach().abs().flatten(1).mean(dim=1)
        scale_x = scale_x + div_mx.detach().abs().flatten(1).mean(dim=1)
        scale_y = dmy_dt.detach().abs().flatten(1).mean(dim=1)
        scale_y = scale_y + div_my.detach().abs().flatten(1).mean(dim=1)
        rx = self.normalize_field(rx, scale_x)
        ry = self.normalize_field(ry, scale_y)
        return rx, ry

    def forward(self, previous_state, next_state, dt, labels=None, step_index=None):
        weight = self.weight("lambda_momentum")
        if weight <= 0.0:
            return next_state.new_zeros(())
        rx, ry = self.residuals(previous_state, next_state, dt)
        loss = 0.5 * (rx.pow(2).mean() + ry.pow(2).mean())
        self.last_components = {"momentum": loss.detach()}
        return weight * loss


class FluidEnergy2D(PhysicsRegularizer):
    """Macroscopic kinetic-energy budget from decoded velocity fields."""

    def velocity(self, state):
        state = self.denormalize(state)
        return self.vector(state, self.cfg("velocity_channels", "velocity"), [1, 2], 2)

    def density(self, state, like):
        if not bool(self.cfg("density_weighted", self.cfg("use_density_weighted_energy", False))):
            return torch.ones_like(like)
        state = self.denormalize(state)
        rho = self.scalar(state, self.cfg("density_channel", "rho"), [0])
        return torch.clamp(rho, min=float(self.cfg("density_floor", 1e-6)))

    def forcing(self, state):
        channels = self.cfg("forcing_channels", None)
        if channels is None:
            return None
        state = self.denormalize(state)
        forcing = self.vector(state, channels, channels, len(_as_list(channels)))
        if forcing.shape[1] == 2:
            return forcing
        if forcing.shape[1] != 1:
            raise ValueError("forcing_channels must identify one scalar or two channels.")
        out = torch.zeros(
            state.shape[0],
            2,
            state.shape[-2],
            state.shape[-1],
            device=state.device,
            dtype=state.dtype,
        )
        out[:, int(self.cfg("forcing_direction", 0))] = forcing[:, 0]
        return out

    def forward(self, previous_state, next_state, dt, labels=None, step_index=None):
        weight = self.weight("lambda_energy", "lambda_fluid_energy", "lambda_kinetic_energy")
        if weight <= 0.0:
            return next_state.new_zeros(())

        previous_velocity = self.velocity(previous_state)
        next_velocity = self.velocity(next_state)
        rho_prev = self.density(previous_state, previous_velocity[:, 0])
        rho_next = self.density(next_state, next_velocity[:, 0])
        cell_area = float(self.cfg("dx", 1.0)) * float(self.cfg("dy", 1.0))

        energy_prev = 0.5 * rho_prev * previous_velocity.pow(2).sum(dim=1)
        energy_next = 0.5 * rho_next * next_velocity.pow(2).sum(dim=1)
        energy_prev = energy_prev.sum(dim=(-2, -1)) * cell_area
        energy_next = energy_next.sum(dim=(-2, -1)) * cell_area

        dudx, dudy = gradient_periodic_2d(
            previous_velocity,
            dx=float(self.cfg("dx", 1.0)),
            dy=float(self.cfg("dy", 1.0)),
        )
        dissipation = (dudx.pow(2).sum(dim=1) + dudy.pow(2).sum(dim=1)).sum(
            dim=(-2, -1)
        ) * cell_area

        forcing = self.forcing(previous_state)
        if forcing is None:
            power = torch.zeros_like(energy_prev)
        else:
            power = (previous_velocity * forcing).sum(dim=1).sum(dim=(-2, -1)) * cell_area

        dt = self.reshape_dt(dt, energy_next)
        budget = (
            energy_next
            - energy_prev
            + dt * float(self.cfg("viscosity", 0.0)) * dissipation
            - dt * power
        )
        mode = self.cfg("mode", self.cfg("energy_mode", "inequality"))
        energy_tau = self.cfg("energy_tau", self.cfg("inequality_tau", 0.0))
        if mode in {"monotone", "decay"}:
            loss = self.positive_part(energy_next - energy_prev, tau=energy_tau).pow(2)
        elif mode in {"inequality", "les", "ineq"}:
            loss = self.positive_part(budget, tau=energy_tau).pow(2)
        elif mode in {"equality", "dns", "eq"}:
            loss = budget.pow(2)
        else:
            raise ValueError("Energy mode must be monotone, inequality/les, or equality/dns.")

        loss = self.normalize_integral_loss(loss, energy_prev, energy_next).mean()
        self.last_components = {"fluid_energy": loss.detach()}
        return weight * loss


class Enstrophy2D(PhysicsRegularizer):
    def vorticity(self, state):
        state = self.denormalize(state)
        channel = self.cfg("vorticity_channel", self.cfg("omega_channel", "omega"))
        try:
            return self.scalar(state, channel, "omega")
        except ValueError:
            velocity = self.vector(state, self.cfg("velocity_channels", "velocity"), [1, 2], 2)
            dvdx, _ = gradient_periodic_2d(
                velocity[:, 1],
                dx=float(self.cfg("dx", 1.0)),
                dy=float(self.cfg("dy", 1.0)),
            )
            _, dudy = gradient_periodic_2d(
                velocity[:, 0],
                dx=float(self.cfg("dx", 1.0)),
                dy=float(self.cfg("dy", 1.0)),
            )
            return dvdx - dudy

    def forward(self, previous_state, next_state, dt, labels=None, step_index=None):
        weight = self.weight("lambda_enstrophy", "lambda_z")
        if weight <= 0.0:
            return next_state.new_zeros(())

        omega_prev = self.vorticity(previous_state)
        omega_next = self.vorticity(next_state)
        cell_area = float(self.cfg("dx", 1.0)) * float(self.cfg("dy", 1.0))
        z_prev = 0.5 * omega_prev.pow(2).sum(dim=(-2, -1)) * cell_area
        z_next = 0.5 * omega_next.pow(2).sum(dim=(-2, -1)) * cell_area
        omega_mid = 0.5 * (omega_prev + omega_next)
        domega_dx, domega_dy = gradient_periodic_2d(
            omega_mid,
            dx=float(self.cfg("dx", 1.0)),
            dy=float(self.cfg("dy", 1.0)),
        )
        dissipation = (domega_dx.pow(2) + domega_dy.pow(2)).sum(dim=(-2, -1)) * cell_area
        dt = self.reshape_dt(dt, z_next)
        budget = z_next - z_prev + dt * float(self.cfg("viscosity", 0.0)) * dissipation

        mode = self.cfg("mode", self.cfg("enstrophy_mode", "inequality"))
        enstrophy_tau = self.cfg("enstrophy_tau", self.cfg("inequality_tau", 0.0))
        if mode in {"monotone", "decay"}:
            loss = self.positive_part(z_next - z_prev, tau=enstrophy_tau).pow(2)
        elif mode in {"inequality", "les", "ineq"}:
            loss = self.positive_part(budget, tau=enstrophy_tau).pow(2)
        elif mode in {"equality", "dns", "eq"}:
            loss = budget.pow(2)
        else:
            raise ValueError("Enstrophy mode must be monotone, inequality/les, or equality/dns.")

        loss = self.normalize_integral_loss(loss, z_prev, z_next).mean()
        self.last_components = {"enstrophy": loss.detach()}
        return weight * loss


class TotalVariationDissipation2D(PhysicsRegularizer):
    def selected_state(self, state):
        state = self.denormalize(state)
        channels = self.cfg_any(("channels", "state_channels", "tvd_channels"), None)
        if channels is None:
            return state
        return state[:, self.resolve_channels(channels, list(range(state.shape[1])))]

    def total_variation(self, state):
        state = self.selected_state(state)
        diff_x = torch.roll(state, shifts=-1, dims=-1) - state
        diff_y = torch.roll(state, shifts=-1, dims=-2) - state
        dx = float(self.cfg("dx", 1.0))
        dy = float(self.cfg("dy", 1.0))
        norm = str(self.cfg("norm", self.cfg("tvd_norm", "anisotropic"))).lower()
        if norm in {"anisotropic", "l1"}:
            density = diff_x.abs() * dy + diff_y.abs() * dx
        elif norm in {"isotropic", "l2"}:
            grad_x = diff_x / max(dx, 1e-12)
            grad_y = diff_y / max(dy, 1e-12)
            eps = float(self.cfg("tvd_eps", 1e-8))
            density = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + eps) * dx * dy
        else:
            raise ValueError("TVD norm must be anisotropic/l1 or isotropic/l2.")
        return density.sum(dim=(1, 2, 3))

    def forward(self, previous_state, next_state, dt, labels=None, step_index=None):
        weight = self.weight("lambda_tvd", "lambda_tv", "lambda_total_variation")
        if weight <= 0.0:
            return next_state.new_zeros(())

        tv_prev = self.total_variation(previous_state)
        tv_next = self.total_variation(next_state)
        tvd_tau = self.cfg("tvd_tau", self.cfg("inequality_tau", 0.0))
        mode = str(self.cfg("mode", self.cfg("tvd_mode", "ratchet"))).lower()
        if mode in {"ratchet", "monotone", "increase"}:
            loss = self.positive_part(tv_next - tv_prev, tau=tvd_tau).pow(2)
        elif mode in {"equality", "eq"}:
            loss = (tv_next - tv_prev).pow(2)
        else:
            raise ValueError("TVD mode must be ratchet/monotone or equality/eq.")

        loss = self.normalize_integral_loss(loss, tv_prev, tv_next).mean()
        self.last_components = {"tvd": loss.detach()}
        return weight * loss


class Divergence2D(PhysicsRegularizer):
    def forward(self, previous_state, next_state, dt, labels=None, step_index=None):
        weight = self.weight("lambda_divergence", "lambda_div")
        if weight <= 0.0:
            return next_state.new_zeros(())
        state = self.denormalize(next_state)
        velocity = self.vector(state, self.cfg("velocity_channels", "velocity"), [1, 2], 2)
        dudx, _ = gradient_periodic_2d(
            velocity[:, 0],
            dx=float(self.cfg("dx", 1.0)),
            dy=float(self.cfg("dy", 1.0)),
        )
        _, dvdy = gradient_periodic_2d(
            velocity[:, 1],
            dx=float(self.cfg("dx", 1.0)),
            dy=float(self.cfg("dy", 1.0)),
        )
        div = dudx + dvdy
        scale = velocity.detach().abs().flatten(1).mean(dim=1)
        div = self.normalize_field(div, scale)
        loss = div.pow(2).mean()
        self.last_components = {"divergence": loss.detach()}
        return weight * loss


class VorticityConsistency2D(PhysicsRegularizer):
    def forward(self, previous_state, next_state, dt, labels=None, step_index=None):
        weight = self.weight("lambda_vorticity_consistency", "lambda_curl")
        if weight <= 0.0:
            return next_state.new_zeros(())
        state = self.denormalize(next_state)
        omega = self.scalar(state, self.cfg("vorticity_channel", "omega"), "omega")
        velocity = self.vector(state, self.cfg("velocity_channels", "velocity"), [1, 2], 2)
        dvdx, _ = gradient_periodic_2d(
            velocity[:, 1],
            dx=float(self.cfg("dx", 1.0)),
            dy=float(self.cfg("dy", 1.0)),
        )
        _, dudy = gradient_periodic_2d(
            velocity[:, 0],
            dx=float(self.cfg("dx", 1.0)),
            dy=float(self.cfg("dy", 1.0)),
        )
        curl = dvdx - dudy
        residual = omega - curl
        scale = omega.detach().abs().flatten(1).mean(dim=1)
        residual = self.normalize_field(residual, scale)
        loss = residual.pow(2).mean()
        self.last_components = {"vorticity_consistency": loss.detach()}
        return weight * loss


class VelocitySpectrum2D(PhysicsRegularizer):
    def __init__(self, config=None, dataset=None):
        super().__init__(config=config, dataset=dataset)
        self._radial_masks = {}

    def radial_masks(self, h, w_rfft, device):
        w = (w_rfft - 1) * 2
        key = (h, w, str(device))
        if key in self._radial_masks:
            return self._radial_masks[key]

        ky = torch.fft.fftfreq(h, device=device) * h
        kx = torch.fft.rfftfreq(w, device=device) * w
        radius = torch.sqrt(ky[:, None].pow(2) + kx[None, :].pow(2))
        max_band = int(min(h, w) // 2)
        masks = [
            (radius >= k - 0.5) & (radius < k + 0.5)
            for k in range(max_band + 1)
        ]
        self._radial_masks[key] = masks
        return masks

    def spectrum(self, state):
        state = self.denormalize(state)
        velocity = self.vector(state, self.cfg("velocity_channels", "velocity"), [1, 2], 2)
        coeff = torch.fft.rfft2(velocity, norm="ortho")
        energy = 0.5 * coeff.abs().pow(2).sum(dim=1)
        h, w_rfft = energy.shape[-2:]
        masks = self.radial_masks(h, w_rfft, energy.device)
        bands = []
        for mask in masks:
            if mask.any():
                bands.append((energy * mask).sum(dim=(-2, -1)))
        return torch.stack(bands, dim=-1)

    def forward(self, previous_state, next_state, dt, labels=None, step_index=None):
        weight = self.weight("lambda_spectrum", "lambda_spec")
        if weight <= 0.0 or labels is None:
            return next_state.new_zeros(())
        pred_spectrum = self.spectrum(next_state)
        true_spectrum = self.spectrum(labels)
        eps = float(self.cfg("spectrum_eps", 1e-12))
        loss = (
            torch.log(pred_spectrum + eps) - torch.log(true_spectrum.detach() + eps)
        ).pow(2).mean()
        self.last_components = {"spectrum": loss.detach()}
        return weight * loss


REGISTRY = {
    "density_positivity": DensityPositivity2D,
    "positivity": DensityPositivity2D,
    "compressible_positivity_2d": CompressiblePositivity2D,
    "pressure_temperature_positivity_2d": CompressiblePositivity2D,
    "pressure_positivity_2d": CompressiblePositivity2D,
    "temperature_positivity_2d": CompressiblePositivity2D,
    "isothermal_euler_entropy_flux_2d": IsothermalEulerEntropyFlux2D,
    "entropy_flux_isothermal_euler_2d": IsothermalEulerEntropyFlux2D,
    "isothermal_euler_entropy_2d": IsothermalEulerEntropyFlux2D,
    "isothermal_euler_2d": IsothermalEulerEntropyFlux2D,
    "burgers_entropy_flux_2d": ScalarBurgersEntropyFlux2D,
    "scalar_burgers_entropy_flux_2d": ScalarBurgersEntropyFlux2D,
    "shallow_water_entropy_flux_2d": ShallowWaterEntropyFlux2D,
    "shallow_water_energy_flux_2d": ShallowWaterEntropyFlux2D,
    "compressible_euler_entropy_flux_2d": CompressibleEulerEntropyFlux2D,
    "ideal_gas_euler_entropy_flux_2d": CompressibleEulerEntropyFlux2D,
    "compressible_ns_entropy_flux_2d": CompressibleNavierStokesEntropyFlux2D,
    "compressible_navier_stokes_entropy_flux_2d": CompressibleNavierStokesEntropyFlux2D,
    "custom_entropy_flux_2d": CustomEntropyFlux2D,
    "entropy_flux_custom_2d": CustomEntropyFlux2D,
    "continuity_2d": ContinuityResidual2D,
    "continuity_residual_2d": ContinuityResidual2D,
    "mass_conservation_2d": ContinuityResidual2D,
    "isothermal_euler_momentum_2d": IsothermalEulerMomentumResidual2D,
    "momentum_residual_2d": IsothermalEulerMomentumResidual2D,
    "fluid_energy_2d": FluidEnergy2D,
    "macroscopic_energy_2d": FluidEnergy2D,
    "macroscopic_kinetic_energy_2d": FluidEnergy2D,
    "kinetic_energy_2d": FluidEnergy2D,
    "incompressible_energy_2d": FluidEnergy2D,
    "incompressible_ns_energy_budget_2d": FluidEnergy2D,
    "incompressible_navier_stokes_energy_budget_2d": FluidEnergy2D,
    "navier_stokes_energy_budget_2d": FluidEnergy2D,
    "enstrophy_2d": Enstrophy2D,
    "vorticity_enstrophy_2d": Enstrophy2D,
    "tvd_2d": TotalVariationDissipation2D,
    "tvd_ratchet_2d": TotalVariationDissipation2D,
    "total_variation_2d": TotalVariationDissipation2D,
    "divergence_2d": Divergence2D,
    "vorticity_consistency_2d": VorticityConsistency2D,
    "velocity_spectrum_2d": VelocitySpectrum2D,
    "spectrum_2d": VelocitySpectrum2D,
}


def _macro_turbulence_regularizers(config):
    entries = []
    if float(config.get("lambda_density", 0.0)) > 0.0:
        entries.append({**config, "kind": "density_positivity", "weight": config["lambda_density"]})
    if float(config.get("lambda_energy", 0.0)) > 0.0:
        entries.append({**config, "kind": "fluid_energy_2d", "weight": config["lambda_energy"]})
    if float(config.get("lambda_enstrophy", 0.0)) > 0.0:
        entries.append({**config, "kind": "enstrophy_2d", "weight": config["lambda_enstrophy"]})
    if float(config.get("lambda_tvd", config.get("lambda_tv", 0.0))) > 0.0:
        entries.append(
            {
                **config,
                "kind": "tvd_2d",
                "weight": config.get("lambda_tvd", config.get("lambda_tv")),
            }
        )
    if float(config.get("lambda_div", 0.0)) > 0.0:
        entries.append({**config, "kind": "divergence_2d", "weight": config["lambda_div"]})
    if float(config.get("lambda_vorticity_consistency", 0.0)) > 0.0:
        entries.append(
            {
                **config,
                "kind": "vorticity_consistency_2d",
                "weight": config["lambda_vorticity_consistency"],
            }
        )
    return entries


def _isothermal_euler_residual_regularizers(config):
    entries = []
    if float(config.get("lambda_continuity", config.get("lambda_mass", 0.0))) > 0.0:
        entries.append({**config, "kind": "continuity_2d"})
    if float(config.get("lambda_momentum", 0.0)) > 0.0:
        entries.append({**config, "kind": "isothermal_euler_momentum_2d"})
    return entries


def _build_one(config, dataset=None):
    if isinstance(config, str):
        config = {"kind": config}
    config = _plain_dict(config) or {}
    if config.get("enabled", True) is False:
        return None
    kind = config.get("kind", config.get("type", None))
    if kind is None:
        raise ValueError("Physics regularizer entries require a 'kind'.")
    kind = str(kind)
    if kind == "macro_turbulence_2d":
        return CompositePhysicsRegularizer(
            [_build_one(entry, dataset=dataset) for entry in _macro_turbulence_regularizers(config)]
        )
    if kind == "isothermal_euler_residual_2d":
        return CompositePhysicsRegularizer(
            [
                _build_one(entry, dataset=dataset)
                for entry in _isothermal_euler_residual_regularizers(config)
            ]
        )
    if kind not in REGISTRY:
        raise ValueError(
            f"Unsupported physics regularizer '{kind}'. "
            f"Available: {', '.join(sorted(REGISTRY))}, macro_turbulence_2d, "
            "isothermal_euler_residual_2d."
        )
    return REGISTRY[kind](config=config, dataset=dataset)


def build_physics_regularizer(config, dataset=None):
    config = _plain_dict(config)
    if config is None or config is False:
        return None
    if isinstance(config, Mapping) and config.get("enabled", True) is False:
        return None

    default_dt_scale = _dataset_time_scale(dataset) if dataset is not None else 1.0
    if isinstance(config, Mapping):
        config = dict(config)
        config.setdefault("dt_scale", default_dt_scale)
        entries = config.get("regularizers", None)
        if entries is not None:
            shared = {k: v for k, v in config.items() if k not in {"regularizers"}}
            regularizers = []
            for entry in entries:
                if isinstance(entry, str):
                    entry = {"kind": entry}
                regularizers.append(_build_one({**shared, **dict(entry)}, dataset=dataset))
            return CompositePhysicsRegularizer(regularizers)
        return _build_one(config, dataset=dataset)

    if isinstance(config, list):
        return CompositePhysicsRegularizer([_build_one(entry, dataset=dataset) for entry in config])

    raise ValueError("physics_regularization must be a mapping or list.")


build_entropy_regularizer = build_physics_regularizer
