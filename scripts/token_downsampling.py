# based on:
#   https://github.com/ethansmith2000/ImprovedTokenMerge
#   https://github.com/ethansmith2000/comfy-todo (MIT)

import math

import gradio as gr
import torch
import torch.nn.functional as F

from modules import scripts, script_callbacks, shared, processing


def up_or_downsample(item, cur_w, cur_h, new_w, new_h, method="nearest-exact"):
    batch_size = item.shape[0]

    item = item.reshape(batch_size, cur_h, cur_w, -1).permute(0, 3, 1, 2)
    item = F.interpolate(item, size=(new_h, new_w), mode=method).permute(0, 2, 3, 1)
    item = item.reshape(batch_size, new_h * new_w, -1)

    return item


def compute_merge(x: torch.Tensor, tome_info: dict):
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))
    cur_h = original_h // downsample
    cur_w = original_w // downsample

    args = tome_info["args"]
    downsample_factor = args["downsample_factor"]

    merge_op = lambda x: x
    if downsample <= args["max_depth"]:
        new_h = int(cur_h / downsample_factor)
        new_w = int(cur_w / downsample_factor)
        merge_op = lambda x: up_or_downsample(x, cur_w, cur_h, new_w, new_h)

    return merge_op


def hook_todo_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args):
        module._todo_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model._todo_info["hooks"].append(model.register_forward_pre_hook(hook))


def hook_attention(attn: torch.nn.Module):
    """ Adds a forward pre hook to downsample attention keys and values. This hook can be removed with remove_patch. """
    def hook(module, args, kwargs):
        hidden_states = args[0]
        m = compute_merge(hidden_states, module._todo_info)
        kwargs["context"] = m(hidden_states)
        return args, kwargs

    attn._todo_info["hooks"].append(attn.register_forward_pre_hook(hook, with_kwargs=True))


def apply_patch(model: torch.nn.Module, downsample_factor: float = 2, max_depth: int = 1):
    """ Patches the UNet's transformer blocks to apply token downsampling. """

    # make sure model isn't already patched
    remove_patch(model)

    diffusion_model = model.model.diffusion_model
    diffusion_model._todo_info = {
        "size": None,
        "hooks": [],
        "args": {
            "downsample_factor": downsample_factor,
            "max_depth": max_depth,
        },
    }
    hook_todo_model(diffusion_model)

    for _, module in diffusion_model.named_modules():
        if module.__class__.__name__ == "BasicTransformerBlock":
            module.attn1._todo_info = diffusion_model._todo_info
            hook_attention(module.attn1)

    return model


def remove_patch(model: torch.nn.Module):
    diffusion_model = model.model.diffusion_model
    if hasattr(diffusion_model, "_todo_info"):
        for hook in diffusion_model._todo_info["hooks"]:
            hook.remove()
        diffusion_model._todo_info["hooks"].clear()

    return model


class TokenDownsamplingScript(scripts.Script):
    def title(self):
        return "Token Downsampling"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p, *args, **kwargs):
        if shared.opts.token_downsampling_factor <= 1:
            remove_patch(shared.sd_model)
            return

        # SDXL only has depth 2 and 3
        max_depth = shared.opts.token_downsampling_max_depth
        if shared.sd_model.is_sdxl and max_depth not in (2, 3):
            max_depth = min(max(max_depth, 2), 3)
            print(f"Token Downsampling: clamped max_depth to {max_depth} for SDXL")

        apply_patch(
            model=shared.sd_model,
            downsample_factor=shared.opts.token_downsampling_factor,
            max_depth=2**(max_depth-1),
        )

        p.extra_generation_params["Token downsampling factor"] = shared.opts.token_downsampling_factor
        p.extra_generation_params["Token downsampling max depth"] = shared.opts.token_downsampling_max_depth


def on_ui_settings():
    section = ("token_downsampling", "Token Downsampling")

    options = {
        "token_downsampling_factor": shared.OptionInfo(
            default=1,
            label="Token downsampling factor",
            component=gr.Slider,
            component_args={"minimum": 1, "maximum": 10, "step": 1},
            infotext="Token downsampling factor",
        ).info("1 = disable"),
        "token_downsampling_max_depth": shared.OptionInfo(
            default=1,
            label="Token downsampling max depth",
            component=gr.Slider,
            component_args={"minimum": 1, "maximum": 4, "step": 1},
            infotext="Token downsampling max depth",
        ).info("Higher affects more layers. For SDXL only 2 and 3 are valid and will be clamped"),
    }

    for name, opt in options.items():
        opt.section = section
        shared.opts.add_option(name, opt)


script_callbacks.on_ui_settings(on_ui_settings)
# TODO: xyz support