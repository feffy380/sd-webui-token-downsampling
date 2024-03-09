# sd-webui-token-downsampling

Implementation of [Token Downsampling](https://arxiv.org/abs/2402.13573) for [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)  
Based on the reference implementation by Ethan Smith: https://github.com/ethansmith2000/ImprovedTokenMerge

**Token Downsampling** is an optimization that improves upon [token merging](https://github.com/dbolya/tomesd), with a focus on improving performance and preserving output quality.

## Usage
Set **Settings > Token Downsampling > Token downsampling factor** above 1 to enable ToDo.  
I recommend downsampling factor 2-4 for SD1.5, a bit lower for SDXL. Use max depth 1 or 2.

## Examples
Coming soon
