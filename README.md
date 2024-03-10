# sd-webui-token-downsampling

Implementation of [Token Downsampling](https://arxiv.org/abs/2402.13573) for [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)  
Based on the reference implementation by Ethan Smith: https://github.com/ethansmith2000/ImprovedTokenMerge

**Token Downsampling** is an optimization that improves upon [token merging](https://github.com/dbolya/tomesd), with a focus on improving performance and preserving output quality.

This extension is compatible with aria1th's [DeepCache extension](https://github.com/aria1th/sd-webui-deepcache-standalone).

## Usage
Settings are found under **Settings > Token Downsampling**.

- **Token downsampling factor**: Set higher than 1 to enable ToDo.
  - Recommended: 2-3
- **Token downsampling max depth**: Raising this affects more layers but reduces quality for not much gain.
  - Recommended: 1 (Default)
- **Token downsampling disable after**: Disable ToDo after a percentage of steps to improve details.
  - Recommended: 0.6-0.8

Downsampling factor and max depth can be raised a bit at higher resolutions (\>1536px) because there's more redundant information.

### LoRA notes

In addition to inference, ToDo (and ToME) can massively speed up training as well. I find that LoRAs trained this way maintain image quality much better at higher downsampling factors, so I can highly recommend training with ToDo if it's an option ([pending PR in kohya/sd-scripts](https://github.com/kohya-ss/sd-scripts/pull/1151)).

## Examples
### Downsample factor vs Max depth
Higher settings greatly reduce quality and don't offer much speed improvement.
![image](https://github.com/feffy380/sd-webui-token-downsampling/assets/114889020/05aa3cd1-9c3a-414f-9936-124e07d01afe)

### Downsample factor vs Disable percentage
Improve detail at higher settings (max depth 2) by disabling ToDo near the end.
![image](https://github.com/feffy380/sd-webui-token-downsampling/assets/114889020/d597db1c-e91d-42cb-b759-e492df963cfe)

### Bonus: [LoRA trained with ToDo](https://github.com/kohya-ss/sd-scripts/pull/1151)
High downsampling factor and max depth with much less quality loss. Combined with DeepCache I get about ~2.3x speedup.
![image](https://github.com/feffy380/sd-webui-token-downsampling/assets/114889020/fd4fe62b-684e-4d40-b4b0-1bec4be0fce0)
