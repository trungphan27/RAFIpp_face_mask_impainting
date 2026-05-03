from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from .loss import LossConfig, LossFactory
from .networks import RAFIpp, PatchDiscriminator, FeaturePatchDiscriminator


class RAFIppSystem(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = RAFIpp()
        self.dp = PatchDiscriminator(in_channels=5)
        self.df = FeaturePatchDiscriminator(in_channels=512)
        loss_cfg = LossConfig(
            lambda_m=args.lambda_m,
            lambda_b=args.lambda_b,
            lambda_c=args.lambda_c,
            lambda_dice=args.lambda_dice,
            lambda_bdice=args.lambda_bdice,
            lambda_rec=args.lambda_rec,
            lambda_ssim=args.lambda_ssim,
            lambda_perc=args.lambda_perc,
            lambda_style=args.lambda_style,
            lambda_id=args.lambda_id,
            lambda_edge=args.lambda_edge,
            lambda_adv=args.lambda_adv,
            alpha_region=args.alpha_region,
            gamma_region=args.gamma_region,
            beta_conf=args.beta_conf,
        )
        self.loss_factory = LossFactory(loss_cfg)

        self.optim_seg = optim.Adam(
            self.model.segnet.parameters(),
            lr=args.lr_seg,
            betas=tuple(args.betas),
            weight_decay=args.weight_decay,
        )
        self.optim_gen = optim.Adam(
            self.model.restore.parameters(),
            lr=args.lr_gen,
            betas=tuple(args.betas),
            weight_decay=args.weight_decay,
        )
        self.optim_joint = optim.Adam(
            list(self.model.segnet.parameters()) + list(self.model.restore.parameters()),
            lr=args.lr_gen,
            betas=tuple(args.betas),
            weight_decay=args.weight_decay,
        )
        self.optim_disc = optim.Adam(
            list(self.dp.parameters()) + list(self.df.parameters()),
            lr=args.lr_disc,
            betas=tuple(args.betas),
            weight_decay=args.weight_decay,
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.loss_factory.to(*args, **kwargs)
        return self

    def set_requires_grad(self, module: nn.Module, flag: bool) -> None:
        for p in module.parameters():
            p.requires_grad = flag

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(batch['masked'])

    def _grad_clip(self, params):
        if self.args.grad_clip and self.args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, self.args.grad_clip)

    def _disc_inputs(self, image, outputs):
        return torch.cat([image, outputs['mask_pred'], outputs['boundary_pred']], dim=1)

    def train_step(self, batch: Dict[str, torch.Tensor], stage: int) -> Dict[str, float]:
        if stage == 1:
            return self._train_stage1(batch)
        if stage == 2:
            return self._train_stage2(batch)
        return self._train_stage3(batch)

    def _train_stage1(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.segnet.train()
        self.model.restore.eval()
        self.dp.eval()
        self.df.eval()

        self.optim_seg.zero_grad(set_to_none=True)
        seg = self.model.segnet(batch['masked'])
        seg_losses = self.loss_factory.segmentation_losses(seg, batch)
        seg_losses['loss_seg'].backward()
        self._grad_clip(self.model.segnet.parameters())
        self.optim_seg.step()
        return {k: float(v.detach().item()) for k, v in seg_losses.items()}

    def _train_stage2(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.segnet.eval()
        self.set_requires_grad(self.model.segnet, False)
        self.model.restore.train()
        self.dp.train()
        self.df.train()

        with torch.no_grad():
            seg = self.model.segnet(batch['masked'])
        restore = self.model.restore(batch['masked'], seg['mask_pred'], seg['boundary_pred'], seg['confidence_pred'])
        outputs = {**seg, **restore}
        rec_losses = self.loss_factory.reconstruction_losses(outputs, batch)

        # Update discriminators.
        self.optim_disc.zero_grad(set_to_none=True)
        real_dp = self.dp(self._disc_inputs(batch['gt'], outputs))
        fake_dp = self.dp(self._disc_inputs(outputs['isyn'].detach(), outputs))
        loss_dp = self.loss_factory.discriminator_hinge(real_dp, fake_dp)
        real_df = self.df(rec_losses['vgg_real_relu4_3'].detach())
        fake_df = self.df(rec_losses['vgg_fake_relu4_3'].detach())
        loss_df = self.loss_factory.discriminator_hinge(real_df, fake_df)
        loss_d = loss_dp + loss_df
        loss_d.backward()
        self._grad_clip(list(self.dp.parameters()) + list(self.df.parameters()))
        self.optim_disc.step()

        # Update generator.
        self.optim_gen.zero_grad(set_to_none=True)
        adv_dp = self.loss_factory.generator_hinge(self.dp(self._disc_inputs(outputs['isyn'], outputs)))
        adv_df = self.loss_factory.generator_hinge(self.df(rec_losses['vgg_fake_relu4_3']))
        adv = adv_dp + adv_df
        seg_losses = {
            'loss_mask': torch.zeros_like(adv),
            'loss_boundary': torch.zeros_like(adv),
            'loss_conf': torch.zeros_like(adv),
            'loss_seg': torch.zeros_like(adv),
        }
        total_losses = self.loss_factory.generator_total(seg_losses, rec_losses, adv, include_seg=False)
        total_losses['loss_total'].backward()
        self._grad_clip(self.model.restore.parameters())
        self.optim_gen.step()

        report = {k: float(v.detach().item()) for k, v in total_losses.items()}
        report['loss_dp'] = float(loss_dp.detach().item())
        report['loss_df'] = float(loss_df.detach().item())
        report['loss_d_total'] = float(loss_d.detach().item())
        return report

    def _train_stage3(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.set_requires_grad(self.model.segnet, True)
        self.dp.train()
        self.df.train()

        # Discriminator update.
        outputs = self.forward(batch)
        rec_losses = self.loss_factory.reconstruction_losses(outputs, batch)
        self.optim_disc.zero_grad(set_to_none=True)
        real_dp = self.dp(self._disc_inputs(batch['gt'], outputs))
        fake_dp = self.dp(self._disc_inputs(outputs['isyn'].detach(), outputs))
        loss_dp = self.loss_factory.discriminator_hinge(real_dp, fake_dp)
        real_df = self.df(rec_losses['vgg_real_relu4_3'].detach())
        fake_df = self.df(rec_losses['vgg_fake_relu4_3'].detach())
        loss_df = self.loss_factory.discriminator_hinge(real_df, fake_df)
        loss_d = loss_dp + loss_df
        loss_d.backward()
        self._grad_clip(list(self.dp.parameters()) + list(self.df.parameters()))
        self.optim_disc.step()

        # Joint generator + segmentation update.
        self.optim_joint.zero_grad(set_to_none=True)
        outputs = self.forward(batch)
        seg_losses = self.loss_factory.segmentation_losses(outputs, batch)
        rec_losses = self.loss_factory.reconstruction_losses(outputs, batch)
        adv_dp = self.loss_factory.generator_hinge(self.dp(self._disc_inputs(outputs['isyn'], outputs)))
        adv_df = self.loss_factory.generator_hinge(self.df(rec_losses['vgg_fake_relu4_3']))
        adv = adv_dp + adv_df
        total_losses = self.loss_factory.generator_total(seg_losses, rec_losses, adv, include_seg=True)
        total_losses['loss_total'].backward()
        self._grad_clip(list(self.model.segnet.parameters()) + list(self.model.restore.parameters()))
        self.optim_joint.step()

        report = {k: float(v.detach().item()) for k, v in total_losses.items()}
        report['loss_dp'] = float(loss_dp.detach().item())
        report['loss_df'] = float(loss_df.detach().item())
        report['loss_d_total'] = float(loss_d.detach().item())
        return report

    @torch.no_grad()
    def inference(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.eval()
        return self.forward(batch)

    def save_checkpoint(self, path: str, epoch: int, stage: int, best_score: float = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'stage': stage,
            'best_score': best_score,
            'model': self.model.state_dict(),
            'dp': self.dp.state_dict(),
            'df': self.df.state_dict(),
            'optim_seg': self.optim_seg.state_dict(),
            'optim_gen': self.optim_gen.state_dict(),
            'optim_joint': self.optim_joint.state_dict(),
            'optim_disc': self.optim_disc.state_dict(),
        }, path)

    def load_checkpoint(self, path: str, map_location='cpu') -> Dict[str, int]:
        ckpt = torch.load(path, map_location=map_location)
        self.model.load_state_dict(ckpt['model'])
        if 'dp' in ckpt:
            self.dp.load_state_dict(ckpt['dp'])
        if 'df' in ckpt:
            self.df.load_state_dict(ckpt['df'])
        if 'optim_seg' in ckpt:
            self.optim_seg.load_state_dict(ckpt['optim_seg'])
        if 'optim_gen' in ckpt:
            self.optim_gen.load_state_dict(ckpt['optim_gen'])
        if 'optim_joint' in ckpt:
            self.optim_joint.load_state_dict(ckpt['optim_joint'])
        if 'optim_disc' in ckpt:
            self.optim_disc.load_state_dict(ckpt['optim_disc'])
        return {
            'epoch': ckpt.get('epoch', 0),
            'stage': ckpt.get('stage', 1),
            'best_score': ckpt.get('best_score', None),
        }
