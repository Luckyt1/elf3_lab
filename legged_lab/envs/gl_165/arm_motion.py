"""GPU playback of checked arm disturbance clips; no AMP expert or discriminator."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from legged_lab.assets.gl_165.model import portable_digest


class ArmMotionDatabase:
    def __init__(self, path, joint_names, nominal, asset_signature, num_envs, device, step_dt):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Missing GL-165 arm motion database: {self.path}")
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with np.load(self.path, allow_pickle=False) as archive:
            self.metadata = json.loads(str(archive["metadata"]))
            if list(archive["joint_names"]) != list(joint_names):
                raise ValueError("Arm database joint order does not match this robot")
            if portable_digest(self.metadata["asset_signature"]) != portable_digest(asset_signature):
                raise ValueError("Arm database is stale: regenerate it for the current GL-165 asset and pose")
            if abs(float(archive["control_dt"]) - step_dt) > 1e-9:
                raise ValueError("Arm database sampling rate does not match the control timestep")
            values = archive["targets"].copy()
        if values.ndim != 3 or values.shape[2] != len(joint_names) or not np.isfinite(values).all():
            raise ValueError("Invalid arm motion database")
        self.nominal = nominal[0].clone()
        self.clips = torch.as_tensor(values, device=device, dtype=nominal.dtype)
        expected = self.nominal.expand(self.clips.shape[0], -1)
        if not (torch.allclose(self.clips[:, 0], expected, atol=1e-6)
                and torch.allclose(self.clips[:, -1], expected, atol=1e-6)):
            raise ValueError("Every arm clip must begin and end at the nominal pose")
        self.clip_ids = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.frame = torch.zeros_like(self.clip_ids)
        self.target = nominal.clone()
        self.reset(torch.arange(num_envs, device=device))

    def reset(self, env_ids):
        self.clip_ids[env_ids] = torch.randint(self.clips.shape[0], (len(env_ids),), device=self.clips.device)
        self.frame[env_ids] = 0
        self.target[env_ids] = self.nominal

    def step(self, scale):
        self.frame += 1
        finished = self.frame >= self.clips.shape[1]
        # No unvalidated splice: every clip has the same stationary endpoint.
        choices = torch.randint(self.clips.shape[0], self.clip_ids.shape, device=self.clips.device)
        self.clip_ids = torch.where(finished, choices, self.clip_ids)
        self.frame.remainder_(self.clips.shape[1])
        self.target = self.nominal + scale * (self.clips[self.clip_ids, self.frame] - self.nominal)
        return self.target
