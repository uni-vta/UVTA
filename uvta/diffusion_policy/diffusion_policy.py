import timm
import torch
from uvta.common.utility.model import freeze_model
from einops import rearrange
from torch import nn


class WorldHead(nn.Module):
    """Regress one future quantity from the (noise-free) conditioning vector.

    Deliberately NOT part of the diffusion target.  Putting the auxiliary blocks
    in the trajectory means one UNet denoises all of them jointly for 16 steps,
    so their residual uncertainty re-enters the action dims at every step.
    Measured over 256 book_teleop anchors, all three models executing the action
    block through an identical unnormalize + rot6d decode:

        target composition      sampled wrist err x/y/z (mm)   step roughness y
        action only             2.58 / 2.81 / 1.26             0.250
        action + tactile        5.96 / 6.18 / 2.56             0.457
        action + state + tactile 7.60 / 9.42 / 4.64            0.873
        (teleop demos, reference)                              0.197

    Note the training loss does NOT show this: on epsilon MSE the three sit at
    2.35 / 3.14 / 3.04 mm, ranking all-on ABOVE action+tactile.  Training loss
    scores one denoising step; deployment runs 16 coupled ones.

    Reading the auxiliary targets off the conditioning keeps their supervision --
    the gradient still flows through ``cond`` into the vision trunk -- while the
    action's reverse process never sees them.  Same split as EgoWAM's separate
    action / world-model heads and Dyna-2's two marginal fields over one trunk.
    """

    def __init__(self, cond_dim: int, horizon: int, dim: int, hidden: int = 512):
        super().__init__()
        self.horizon = int(horizon)
        self.dim = int(dim)
        self.net = nn.Sequential(
            nn.Linear(int(cond_dim), int(hidden)),
            nn.Mish(),
            nn.Linear(int(hidden), self.horizon * self.dim),
        )

    def forward(self, condition):
        return self.net(condition).view(-1, self.horizon, self.dim)


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        vision_backbone_kwargs,
        freeze_vision_back,
        diffusion_policy_head,
        world_heads: dict | None = None,
        world_head_weights: dict | None = None,
    ):
        super().__init__()
        # self.vision_backbone = timm.create_model(
        #     **vision_backbone_kwargs, pretrained=True
        # )
        self.vision_backbone = timm.create_model(
            **vision_backbone_kwargs, pretrained=False
        )
        self.freeze_vision_back = freeze_vision_back
        if self.freeze_vision_back:
            freeze_model(self.vision_backbone)
        self.vision_backbone.train()
        self.diffusion_policy_head = diffusion_policy_head
        # ``world_heads``: {name: WorldHead}.  Empty (the default) reproduces the
        # action-only model exactly -- no new parameters, no new loss terms.
        self.world_heads = nn.ModuleDict(world_heads or {})
        self.world_head_weights = dict(world_head_weights or {})
        if self.world_heads:
            print(
                "[DiffusionPolicy] world heads: "
                + ", ".join(
                    f"{n}({h.horizon}x{h.dim}, w={self.world_head_weights.get(n, 1.0)})"
                    for n, h in self.world_heads.items()
                )
                + f"  | +{sum(p.numel() for p in self.world_heads.parameters())/1e6:.2f}M params"
            )

    def forward(
        self,
        noisy_actions,
        timesteps,
        proprioception,
        fsr,
        visual_obs,
        noise,
        return_noise_pred: bool = False,
        aux_targets: dict | None = None,
    ):
        condition = self.encode_condition(proprioception, fsr, visual_obs)
        noise_prediction = self.diffusion_policy_head(
            noisy_actions, timesteps, global_cond=condition
        )
        # diffusion loss
        loss = nn.functional.mse_loss(noise_prediction, noise)

        # Auxiliary world-head losses.  They share the trunk through
        # ``condition`` and so still shape the visual representation, but they are
        # absent from the diffusion target, so they cannot perturb the action's
        # reverse process.
        aux_losses = {}
        if self.world_heads:
            missing = set(self.world_heads) - set(aux_targets or {})
            if missing:
                raise ValueError(
                    f"world heads {sorted(missing)} have no target in this batch; "
                    "the dataset must run with aux_as_head=True so it emits "
                    "aux_<name> keys."
                )
            for name, head in self.world_heads.items():
                aux_losses[name] = nn.functional.mse_loss(
                    head(condition), aux_targets[name]
                )
                loss = loss + float(
                    self.world_head_weights.get(name, 1.0)
                ) * aux_losses[name]

        if return_noise_pred:
            # Caller wants the per-element residual so it can split the loss
            # into sub-action-dim parts (e.g. eef vs hand-joint MSE), plus the
            # auxiliary terms for logging.  ``loss`` is the backprop scalar.
            return loss, noise_prediction, aux_losses
        return loss

    def encode_condition(self, proprioception, fsr, visual_obs):
        """Vision + proprioception + tactile -> the global conditioning vector."""
        bsz, obs_horizon, C, H, W = visual_obs.shape
        visual_obs = rearrange(visual_obs, "b o c h w -> (b o) c h w")
        visual_embedding = self.vision_backbone(visual_obs)  # (bsz * obs_horizon, D)
        visual_embedding = rearrange(
            visual_embedding, "(b o) d -> b (o d)", b=bsz, o=obs_horizon
        )
        condition = [visual_embedding]
        if proprioception is not None:
            condition.append(proprioception.flatten(start_dim=1))
        if fsr is not None:
            condition.append(fsr.flatten(start_dim=1))
        return torch.cat(condition, dim=1)

    @torch.no_grad()
    def predict_world(self, proprioception, fsr, visual_obs):
        """Run the world heads only -- one forward pass, no diffusion.

        For the two-stage rollout, which consumes the predicted future state and
        tactile.  Note these are MSE regressions, i.e. the conditional MEAN, not a
        sample: deterministic and smooth, but mode-averaging where the future is
        genuinely multimodal.  Swap a head for a flow-matching one if sampling
        matters more than smoothness.
        """
        if not self.world_heads:
            return {}
        condition = self.encode_condition(proprioception, fsr, visual_obs)
        return {n: h(condition) for n, h in self.world_heads.items()}

    @staticmethod
    def _profile_cuda_event(enabled):
        if not enabled or not torch.cuda.is_available():
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def condition_sample(
        self,
        cond,
        trajectory,
        noise_scheduler,
        inference_profile=None,
    ):
        if inference_profile is None:
            for t in noise_scheduler.timesteps:
                with torch.no_grad():
                    model_output = self.diffusion_policy_head(
                        trajectory, t.unsqueeze(0).cuda(), cond
                    )
                trajectory = noise_scheduler.step(
                    model_output, t, trajectory
                ).prev_sample
            return trajectory

        step_events = []
        for t in noise_scheduler.timesteps:
            step_start = self._profile_cuda_event(inference_profile is not None)
            with torch.no_grad():
                model_output = self.diffusion_policy_head(
                    trajectory, t.unsqueeze(0).cuda(), cond
                )
            trajectory = noise_scheduler.step(model_output, t, trajectory).prev_sample
            step_end = self._profile_cuda_event(inference_profile is not None)
            if step_start is not None and step_end is not None:
                step_events.append((step_start, step_end))

        if inference_profile is not None:
            inference_profile["_denoise_step_cuda_events"] = step_events

        return trajectory

    def inference(
        self,
        proprioception,
        fsr,
        visual_obs,
        trajectory,
        noise_scheduler,
        num_inference_steps,
        inference_profile=None,
    ):
        profile_enabled = inference_profile is not None
        total_start = self._profile_cuda_event(profile_enabled)
        noise_scheduler.set_timesteps(num_inference_steps)
        bsz, obs_horizon, C, H, W = visual_obs.shape
        visual_obs = rearrange(visual_obs, "b o c h w -> (b o) c h w")
        vision_start = self._profile_cuda_event(profile_enabled)
        visual_embedding = self.vision_backbone(visual_obs)  # (bsz * obs_horizon, D)
        vision_end = self._profile_cuda_event(profile_enabled)
        visual_embedding = rearrange(
            visual_embedding, "(b o) d -> b (o d)", b=bsz, o=obs_horizon
        )
        condition = [visual_embedding]
        if proprioception is not None:
            proprioception = rearrange(proprioception, "b o c -> b (o c)")
            condition.append(proprioception)
        if fsr is not None:
            fsr = rearrange(fsr, "b o c -> b (o c)")
            condition.append(fsr)

        condition = torch.cat(condition, dim=1)
        trajectory = self.condition_sample(
            condition,
            trajectory,
            noise_scheduler,
            inference_profile=inference_profile,
        )
        total_end = self._profile_cuda_event(profile_enabled)
        if inference_profile is not None:
            inference_profile["_diffusion_total_cuda_events"] = (
                total_start,
                total_end,
            )
            inference_profile["_vision_cuda_events"] = (
                vision_start,
                vision_end,
            )
        return trajectory
