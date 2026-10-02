import os
import gc
import json
import torch
from typing import Dict, Any

import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar
from comfy.ldm.colormap import turbo as _turbo

current_dir = os.path.dirname(os.path.abspath(__file__))

DEFAULT_LOTUS_REPO = "1038lab/Lotus-2"

LOTUS_PRESET_MODELS = [
    "lotus-depth-g-v2-1-disparity-fp16.safetensors",
    "lotus-normal-g-v1-1-fp16.safetensors",
]

# Enable hardware Tensor Core and CuDNN acceleration for Ampere/Ada (RTX 30xx/40xx)
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def clean_vram():
    """Safely clear cached VRAM across CUDA, MPS, and CPU backends."""
    gc.collect()
    mm.soft_empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass



def _load_state_dict_any(path: str):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def ensure_lotus_model(model_name: str) -> str:
    """Ensure standalone Lotus safetensors model (1038lab/Lotus-2) exists locally."""
    p = folder_paths.get_full_path("geometry_estimation", model_name)
    if p is not None and os.path.exists(p):
        return p

    model_dirs = folder_paths.get_folder_paths("geometry_estimation")
    target_dir = model_dirs[0] if model_dirs else os.path.join(folder_paths.models_dir, "geometry_estimation")
    target_path = os.path.join(target_dir, model_name)

    if not os.path.exists(target_path):
        os.makedirs(target_dir, exist_ok=True)
        print(f"[Lotus] Downloading {model_name} from {DEFAULT_LOTUS_REPO} to {target_dir}...")
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=DEFAULT_LOTUS_REPO,
            filename=model_name,
            local_dir=target_dir,
        )
        if os.path.exists(downloaded):
            return downloaded

    return target_path


def load_vae(dtype: torch.dtype, device: torch.device):
    """Load SD VAE: check local models/vae first with local offline config (zero HTTP requests)."""
    from diffusers.models import AutoencoderKL

    vae_config_path = os.path.join(current_dir, "configs", "lotus_vae_config.json")

    # 1. Search for any standard SD VAE in ComfyUI VAE directories (e.g. models/vae/)
    local_vaes = folder_paths.get_filename_list("vae")
    matched_vae = None
    for f in local_vaes:
        f_lower = f.lower()
        if "ft-mse" in f_lower or "sd-vae" in f_lower or "sd2" in f_lower:
            matched_vae = f
            break

    if not matched_vae:
        for f in local_vaes:
            if f.lower().endswith(".safetensors"):
                matched_vae = f
                break

    if matched_vae:
        vae_path = folder_paths.get_full_path("vae", matched_vae)
        if vae_path and os.path.exists(vae_path):
            print(f"[Lotus] Using local VAE: {vae_path}")
            return AutoencoderKL.from_single_file(vae_path, config=vae_config_path, torch_dtype=dtype).to(device)

    # 2. If missing, auto-download standalone VAE safetensors into models/vae
    vae_dirs = folder_paths.get_folder_paths("vae")
    target_dir = vae_dirs[0] if vae_dirs else os.path.join(folder_paths.models_dir, "vae")
    target_path = os.path.join(target_dir, "vae-ft-mse-840000-ema-pruned.safetensors")

    if not os.path.exists(target_path):
        os.makedirs(target_dir, exist_ok=True)
        print(f"[Lotus] VAE not found locally. Downloading vae-ft-mse-840000-ema-pruned.safetensors to {target_dir}...")
        from huggingface_hub import hf_hub_download
        try:
            downloaded = hf_hub_download(
                repo_id=DEFAULT_LOTUS_REPO,
                filename="vae-ft-mse-840000-ema-pruned.safetensors",
                local_dir=target_dir,
            )
            target_path = downloaded
        except Exception:
            downloaded = hf_hub_download(
                repo_id="stabilityai/sd-vae-ft-mse",
                filename="vae-ft-mse-840000-ema-pruned.safetensors",
                local_dir=target_dir,
            )
            target_path = downloaded

    return AutoencoderKL.from_single_file(target_path, config=vae_config_path, torch_dtype=dtype).to(device)


# ============================================================================
# Standalone Lotus Pipeline (UNet-based, ~1.7GB, Fast & Light)
# ============================================================================

_LOTUS_CACHE: Dict[str, Any] = {}


def load_lotus_pipeline(model_path: str, dtype: torch.dtype, device: torch.device):
    cache_key = f"{model_path}|{dtype}|{device}"
    if cache_key in _LOTUS_CACHE:
        return _LOTUS_CACHE[cache_key]

    from diffusers.models import UNet2DConditionModel

    print(f"[Lotus] Loading UNet model from {model_path}...")
    if model_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        try:
            lotus_sd = load_file(model_path, device=str(device))
        except Exception:
            lotus_sd = load_file(model_path)
    else:
        lotus_sd = _load_state_dict_any(model_path)

    in_channels = lotus_sd["conv_in.weight"].shape[1]

    config_path = os.path.join(current_dir, "configs", "lotus_unet_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    config_data["in_channels"] = in_channels

    # Instant zero-overhead meta initialization: bypasses 40s of CPU weight generation
    with torch.device("meta"):
        unet = UNet2DConditionModel.from_config(config_data)

    unet.load_state_dict(lotus_sd, assign=True)
    unet.to(device=device, dtype=dtype)
    unet.eval()

    vae = load_vae(dtype, device)
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    vae.eval()

    prompt_path = os.path.join(current_dir, "configs", "empty_text_embed.pt")
    prompt_embeds = torch.load(prompt_path, weights_only=True).to(device=device, dtype=dtype)

    cached = {
        "unet": unet,
        "vae": vae,
        "prompt_embeds": prompt_embeds,
        "in_channels": in_channels,
        "dtype": dtype,
        "device": device,
        "cache_key": cache_key,
    }
    _LOTUS_CACHE[cache_key] = cached
    return cached

_SPECTRAL_ANCHORS = [
    (0.6196078431, 0.0039215686, 0.2588235294),
    (0.8352941176, 0.2431372549, 0.3098039216),
    (0.9568627451, 0.4274509804, 0.2627450980),
    (0.9921568627, 0.6823529412, 0.3803921569),
    (0.9960784314, 0.8784313725, 0.5450980392),
    (1.0000000000, 1.0000000000, 0.7490196078),
    (0.9019607843, 0.9607843137, 0.5960784314),
    (0.6705882353, 0.8666666667, 0.6431372549),
    (0.4000000000, 0.7607843137, 0.6470588235),
    (0.1960784314, 0.5333333333, 0.7411764706),
    (0.3686274510, 0.3098039216, 0.6352941176),
]

_INFERNO_ANCHORS = [
    (0.001462, 0.000466, 0.013866),
    (0.087411, 0.044556, 0.224813),
    (0.258234, 0.038571, 0.406485),
    (0.441788, 0.081108, 0.428009),
    (0.626270, 0.154454, 0.357919),
    (0.798216, 0.280197, 0.243538),
    (0.928329, 0.472975, 0.086005),
    (0.983058, 0.697444, 0.089408),
    (0.965416, 0.915014, 0.354746),
    (0.988362, 0.998364, 0.644924),
]


def _build_lut(anchors_list, n=256):
    anchors = torch.tensor(anchors_list, dtype=torch.float32)
    xs = torch.linspace(0.0, 1.0, n)
    seg = len(anchors_list) - 1
    pos = xs * seg
    lo = pos.floor().clamp(0, seg - 1).long()
    frac = (pos - lo).unsqueeze(-1)
    return anchors[lo] * (1 - frac) + anchors[lo + 1] * frac


_SPECTRAL_LUT = _build_lut(_SPECTRAL_ANCHORS, 256)
_INFERNO_LUT = _build_lut(_INFERNO_ANCHORS, 256)


class Lotus:
    """All-in-one Lotus Node for SOTA Geometric Estimation (Depth & Surface Normal)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Input video frames or single image. Connect from Load Image or Load Video.",
                    },
                ),
                "mode": (
                    ["depth", "normal"],
                    {
                        "default": "depth",
                        "tooltip": (
                            "Geometry estimation mode:\n"
                            "- depth: SOTA Disparity Depth map\n"
                            "- normal: SOTA Surface Normal vector map"
                        ),
                    },
                ),
                "resolution": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Model internal processing resolution. Automatically resized back to input resolution. Set to 768 or 512 for 2x~3x faster video processing!",
                    },
                ),
                "normalization": (
                    ["min_max", "raw"],
                    {
                        "default": "min_max",
                        "tooltip": (
                            "Depth range normalization mode:\n"
                            "- min_max: Standard 0 to 1 range (near=white 1.0, far=black 0.0)\n"
                            "- raw: Unscaled numerical depth values"
                        ),
                    },
                ),
                "colormap": (
                    ["gray", "spectral", "turbo", "inferno"],
                    {
                        "default": "gray",
                        "tooltip": (
                            "Output color format for depth:\n"
                            "- gray: Grayscale (Required for ControlNet and Depth-to-Video)\n"
                            "- spectral: Lotus official thermal colormap\n"
                            "- turbo: Rainbow colormap\n"
                            "- inferno: High-contrast thermal heatmap"
                        ),
                    },
                ),
                "weight_dtype": (
                    ["fp16", "bf16", "fp32"],
                    {
                        "default": "fp16",
                        "tooltip": "Computation precision: fp16 (recommended for speed & VRAM), bf16, or fp32.",
                    },
                ),
                "unload_model": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Unload model from VRAM immediately after execution to free maximum GPU memory.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "🧪AILab/Geometry"

    def process(
        self,
        image,
        mode="depth",
        resolution=1024,
        normalization="min_max",
        colormap="gray",
        weight_dtype="fp16",
        unload_model=False,
    ):
        model_name = (
            "lotus-depth-g-v2-1-disparity-fp16.safetensors"
            if mode == "depth"
            else "lotus-normal-g-v1-1-fp16.safetensors"
        )
        model_path = ensure_lotus_model(model_name)
        device = mm.get_torch_device()
        target_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(weight_dtype, torch.float16)

        cached = load_lotus_pipeline(model_path, target_dtype, device)
        unet = cached["unet"]
        vae = cached["vae"]
        in_channels = cached["in_channels"]
        prompt_embeds = cached["prompt_embeds"]

        B, H_orig, W_orig, C = image.shape
        pbar = ProgressBar(B)

        # Scale to resolution (multiple of 64)
        max_edge = max(H_orig, W_orig)
        scale = resolution / max(max_edge, 1)
        new_h = max(int((int(H_orig * scale) // 64) * 64), 64)
        new_w = max(int((int(W_orig * scale) // 64) * 64), 64)

        timesteps = torch.tensor(999, device=device).long()
        task_emb = torch.tensor([1, 0], device=device, dtype=target_dtype).unsqueeze(0)
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)

        decoded_frames = []

        try:
            with torch.inference_mode():
                for i in range(B):
                    # Process frame-by-frame to prevent VRAM explosion and freezing on videos
                    frame = image[i:i + 1, ..., :3].permute(0, 3, 1, 2).contiguous().float()
                    frame_resized = torch.nn.functional.interpolate(frame, size=(new_h, new_w), mode="bilinear", align_corners=False)
                    frame_in = (frame_resized * 2.0 - 1.0).to(device=device, dtype=target_dtype)

                    latents = vae.encode(frame_in).latent_dist.mode() * 0.18215

                    torch.manual_seed(0)
                    if in_channels == 8:
                        noise = torch.randn(latents.shape, device=device, dtype=target_dtype)
                        latents_in = torch.cat([latents, noise], dim=1)
                    else:
                        latents_in = latents

                    sub_latents = unet(
                        latents_in,
                        timesteps,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=None,
                        return_dict=False,
                        class_labels=task_emb,
                    )[0]

                    sub_latents = sub_latents / 0.18215
                    dec = vae.decode(sub_latents, return_dict=False)[0]

                    # Scale back to original resolution and immediately move to CPU memory
                    dec = (dec / 2.0 + 0.5).clamp(0.0, 1.0)
                    dec = torch.nn.functional.interpolate(dec, size=(H_orig, W_orig), mode="bilinear", align_corners=False)
                    frame_out = dec.permute(0, 2, 3, 1).contiguous().float().cpu()
                    decoded_frames.append(frame_out)

                    pbar.update(1)

                    if B > 1 and (i + 1) % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            if unload_model:
                cached_item = _LOTUS_CACHE.pop(cached["cache_key"], None)
                if cached_item:
                    cached_item["unet"].to("cpu")
                    cached_item["vae"].to("cpu")
                    del cached_item
                mm.unload_all_models()
                clean_vram()
            else:
                clean_vram()

        out = torch.cat(decoded_frames, dim=0)

        is_normal = (mode == "normal")

        if not is_normal:
            mono = out.mean(dim=-1, keepdim=True)
            if normalization == "min_max":
                out_depth = []
                for i in range(B):
                    m = mono[i]
                    mn, mx = m.min(), m.max()
                    norm_m = (m - mn) / max(mx - mn, 1e-6)
                    out_depth.append(norm_m)
                norm = torch.stack(out_depth, dim=0)
            else:
                norm = mono

            gray_output = norm.repeat(1, 1, 1, 3)

            if colormap == "spectral":
                lut = _SPECTRAL_LUT.to(norm.device)
                d = 1.0 - norm.squeeze(-1).clamp(0.0, 1.0)
                idx = (d * 255.0).round().clamp(0, 255).long()
                output = lut[idx]
            elif colormap == "turbo":
                output = _turbo(norm.squeeze(-1).clamp(0.0, 1.0))
            elif colormap == "inferno":
                lut = _INFERNO_LUT.to(norm.device)
                d = norm.squeeze(-1).clamp(0.0, 1.0)
                idx = (d * 255.0).round().clamp(0, 255).long()
                output = lut[idx]
            else:
                output = gray_output

            return (output.contiguous().float(),)
        else:
            return (out.contiguous().float(),)


NODE_CLASS_MAPPINGS = {
    "Lotus": Lotus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Lotus": "Lotus",
}
