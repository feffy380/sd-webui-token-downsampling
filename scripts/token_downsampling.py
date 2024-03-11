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
    cur_h = math.ceil(original_h / downsample)
    cur_w = math.ceil(original_w / downsample)

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
        module._todo_info["current_step"] += 1
        return None

    model._todo_info["hooks"].append(model.register_forward_pre_hook(hook))


def hook_attention(attn: torch.nn.Module):
    """ Adds a forward pre hook to downsample attention keys and values. This hook can be removed with remove_patch. """
    def hook(module, args, kwargs):
        if module._todo_info["current_step"] > module._todo_info["args"]["disable_after"]:
            return
        hidden_states = args[0]
        m = compute_merge(hidden_states, module._todo_info)
        kwargs["context"] = m(hidden_states)
        return args, kwargs

    attn._todo_info["hooks"].append(attn.register_forward_pre_hook(hook, with_kwargs=True))


def apply_patch(model: torch.nn.Module, downsample_factor: float = 2, max_depth: int = 1, disable_after: float = 1.0):
    """ Patches the UNet's transformer blocks to apply token downsampling. """

    # make sure model isn't already patched
    remove_patch(model)

    diffusion_model = model.model.diffusion_model
    diffusion_model._todo_info = {
        "size": None,
        "current_step": 0,
        "hooks": [],
        "args": {
            "downsample_factor": downsample_factor,
            "max_depth": max_depth,
            "disable_after": disable_after,
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

    def _enabled(self, p):
        return getattr(p, "token_downsampling_factor", shared.opts.token_downsampling_factor) > 1

    def process(self, p, *args, **kwargs):
        if not self._enabled(p):
            remove_patch(shared.sd_model)
            return

        # xyz overrides settings via p
        downsample_factor = getattr(p, "token_downsampling_factor", shared.opts.token_downsampling_factor)
        max_depth = getattr(p, "token_downsampling_max_depth", shared.opts.token_downsampling_max_depth)
        disable_after = getattr(p, "token_downsampling_disable_after", shared.opts.token_downsampling_disable_after)

        # SDXL only has depth 2 and 3
        if shared.sd_model.is_sdxl and max_depth not in (2, 3):
            max_depth = min(max(max_depth, 2), 3)
            print(f"Token Downsampling: clamped max_depth to {max_depth} for SDXL")

        apply_patch(
            model=shared.sd_model,
            downsample_factor=downsample_factor,
            max_depth=2**(max_depth-1),
            disable_after=int(disable_after * p.steps),
        )

        p.extra_generation_params["ToDo factor"] = downsample_factor
        p.extra_generation_params["ToDo max depth"] = max_depth
        p.extra_generation_params["ToDo disable after"] = disable_after

    def process_batch(self, p, *args, **kwargs):
        if self._enabled(p):
            shared.sd_model.model.diffusion_model._todo_info["current_step"] = 0


def on_ui_settings():
    section = ("token_downsampling", "Token Downsampling")

    options = {
        "token_downsampling_factor": shared.OptionInfo(
            default=1.0,
            label="Token downsampling factor",
            component=gr.Slider,
            component_args={"minimum": 1.0, "maximum": 5.0, "step": 0.5},
            infotext="ToDo factor",
        ).info("1 = disable"),
        "token_downsampling_max_depth": shared.OptionInfo(
            default=1,
            label="Token downsampling max depth",
            component=gr.Slider,
            component_args={"minimum": 1, "maximum": 4, "step": 1},
            infotext="ToDo max depth",
        ).info("Higher affects more layers. For SDXL only 2 and 3 are valid and will be clamped"),
        "token_downsampling_disable_after": shared.OptionInfo(
            default=1.0,
            label="Token downsampling disable after",
            component=gr.Slider,
            component_args={"minimum": 0.0, "maximum": 1.0, "step": 0.1},
            infotext="ToDo disable after",
        ).info("Disable ToDo after a percentage of steps to improve details"),
    }

    for name, opt in options.items():
        opt.section = section
        shared.opts.add_option(name, opt)


def add_xyz_axis_options():
    xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module

    todo_axis_options = [
        xyz_grid.AxisOption("[ToDo] Downsampling factor", float, xyz_grid.apply_field("token_downsampling_factor")),
        xyz_grid.AxisOption("[ToDo] Max depth", int, xyz_grid.apply_field("token_downsampling_max_depth"), choices=lambda: [1, 2, 3, 4]),
        xyz_grid.AxisOption("[ToDo] Disable after", float, xyz_grid.apply_field("token_downsampling_disable_after")),
    ]

    xyz_grid.axis_options.extend(todo_axis_options)


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_before_ui(add_xyz_axis_options)
