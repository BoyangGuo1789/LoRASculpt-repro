import torch

from llava.train.TPSAMIX_Trainer import TPSAMIX


class TFRLORA(TPSAMIX):
    """Target-Frozen Residual LoRA trainer.

    The adapter is initialized as a rank-expanded target LoRA:
    the first rank block stores the reproduced target adapter and the second
    block starts as a zero-function residual branch. This trainer freezes the
    target block by masking its gradients, so source/anchor training can only
    move the residual ranks while inference remains a single always-on adapter.
    """

    def _tfr_freeze_rank(self):
        return int(getattr(self.args, "tfr_freeze_rank", 0) or 0)

    def _install_tfr_hooks(self, model):
        if getattr(self, "_tfr_hooks_installed", False):
            return

        freeze_rank = self._tfr_freeze_rank()
        if freeze_rank <= 0:
            self._tfr_hooks_installed = True
            return

        hook_count = 0
        frozen_values = 0
        trainable_values = 0
        self._tfr_frozen_state = {}
        for name, param in self._unwrap_model(model).named_parameters():
            if not param.requires_grad or ".teacher." in name:
                continue
            if "lora_A" not in name and "lora_B" not in name:
                continue
            if param.dim() != 2:
                continue

            mask = torch.ones_like(param, dtype=torch.float32, device=param.device)
            if "lora_A" in name:
                rows = min(freeze_rank, param.shape[0])
                mask[:rows, :] = 0.0
                self._tfr_frozen_state[name] = param.detach()[:rows, :].clone()
            elif "lora_B" in name:
                cols = min(freeze_rank, param.shape[1])
                mask[:, :cols] = 0.0
                self._tfr_frozen_state[name] = param.detach()[:, :cols].clone()

            frozen_values += int(mask.numel() - mask.count_nonzero().item())
            trainable_values += int(mask.count_nonzero().item())
            param.register_hook(lambda grad, mask=mask: grad * mask.to(device=grad.device, dtype=grad.dtype))
            hook_count += 1

        if hook_count == 0:
            raise ValueError("TFR-LoRA found no trainable LoRA parameters to mask")

        self._tfr_hooks_installed = True
        if getattr(self.args, "local_rank", -1) in (-1, 0):
            print(
                f"[TFR-LoRA] installed gradient masks: hooks={hook_count} "
                f"freeze_rank={freeze_rank} frozen_values={frozen_values} "
                f"residual_trainable_values={trainable_values}"
            )

    def restore_tfr_frozen_blocks(self, model=None):
        if model is None:
            model = self.model
        state = getattr(self, "_tfr_frozen_state", None)
        if not state:
            return

        with torch.no_grad():
            for name, param in self._unwrap_model(model).named_parameters():
                ref = state.get(name)
                if ref is None:
                    continue
                if ref.device != param.device or ref.dtype != param.dtype:
                    ref = ref.to(device=param.device, dtype=param.dtype)
                if "lora_A" in name:
                    param[: ref.shape[0], :].copy_(ref)
                elif "lora_B" in name:
                    param[:, : ref.shape[1]].copy_(ref)

    def _lora_l2_anchor_loss(self, model):
        weight = float(getattr(self.args, "tfr_residual_l2", 0.0) or 0.0)
        if weight <= 0.0:
            return self._zero(model)

        freeze_rank = self._tfr_freeze_rank()
        if freeze_rank <= 0:
            return self._zero(model)

        loss = self._zero(model)
        count = 0
        for name, param in self._unwrap_model(model).named_parameters():
            if ".teacher." in name:
                continue
            if "lora_A" in name and param.dim() == 2 and freeze_rank < param.shape[0]:
                residual = param[freeze_rank:, :]
            elif "lora_B" in name and param.dim() == 2 and freeze_rank < param.shape[1]:
                residual = param[:, freeze_rank:]
            else:
                continue
            loss = loss + residual.float().pow(2).mean()
            count += 1
        if count == 0:
            return self._zero(model)
        return weight * loss / count

    def compute_loss(self, model, inputs, return_outputs=False):
        self._install_tfr_hooks(model)
        self.restore_tfr_frozen_blocks(model)
        return super().compute_loss(model, inputs, return_outputs=return_outputs)
