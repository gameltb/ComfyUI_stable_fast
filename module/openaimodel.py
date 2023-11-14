import torch as th
import torch.nn as nn
import copy

from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel,forward_timestep_embed,apply_control
from comfy.ldm.modules.diffusionmodules.util import timestep_embedding


class PatchUNetModel(UNetModel):

    @staticmethod
    def cast_from(other):
        if isinstance(other, UNetModel):
            other.__class__ = PatchUNetModel
            other.patch_init()
            return other
        raise ValueError(f"instance must be comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel")

    def cast_to_base_model(self):
        self.patch_deinit()
        self.__class__ = UNetModel
        return self

    def patch_init(self):
        self.output_block_patch = nn.ModuleList()

    def patch_deinit(self):
        del self.output_block_patch

    def set_output_block_patch(self, output_block_patch):
        self.output_block_patch = nn.ModuleList([nn.ModuleList(copy.deepcopy(output_block_patch)) for _ in self.output_blocks])

    def forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["current_index"] = 0
        transformer_patches = transformer_options.get("patches", {})

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(self.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for id, module in enumerate(self.input_blocks):
            transformer_options["block"] = ("input", id)
            h = forward_timestep_embed(module, h, emb, context, transformer_options)
            h = apply_control(h, control, 'input')
            hs.append(h)

        transformer_options["block"] = ("middle", 0)
        h = forward_timestep_embed(self.middle_block, h, emb, context, transformer_options)
        h = apply_control(h, control, 'middle')

        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            hsp = apply_control(hsp, control, 'output')

            for patch_id ,output_block_patch_module in enumerate(self.output_block_patch[id]):
                h, hsp = output_block_patch_module(h, hsp, transformer_patches.get("output_block_patch")[patch_id])
                # h, hsp = output_block_patch_module(h, hsp, transformer_options)

            # if "output_block_patch" in transformer_patches:
            #     patch = transformer_patches["output_block_patch"]
            #     for p in patch:
            #         h, hsp = p(h, hsp, transformer_options)

            h = th.cat([h, hsp], dim=1)
            del hsp
            if len(hs) > 0:
                output_shape = hs[-1].shape
            else:
                output_shape = None
            h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
