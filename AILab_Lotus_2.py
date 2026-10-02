import os
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

import folder_paths
import comfy.model_management as mm
import comfy.sd
import comfy.utils

log = logging.getLogger("Lotus2")
current_dir = os.path.dirname(os.path.abspath(__file__))

HF_REPO = "1038lab/Lotus-2"
ALT_HF_REPO = "jingheya/Lotus-2"

FILES = {
    "depth": {
        "core": "lotus-2_core_predictor_depth.safetensors",
        "sharpener": "lotus-2_detail_sharpener_depth.safetensors",
        "lcm": "lotus-2_lcm_depth.safetensors",
    },
    "normal": {
        "core": "lotus-2_core_predictor_normal.safetensors",
        "sharpener": "lotus-2_detail_sharpener_normal.safetensors",
        "lcm": "lotus-2_lcm_normal.safetensors",
    },
}

BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 4096
BASE_SHIFT = 0.5
MAX_SHIFT = 1.15
NUM_TRAIN_TIMESTEPS = 10


# ---------------------------------------------------------------------------
# LCM Architecture
# ---------------------------------------------------------------------------
class LocalContinuityModule(nn.Module):
    def __init__(self, num_channels=16):
        super().__init__()
        self.lcm = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_channels * 2, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        lcm_dtype = next(self.lcm.parameters()).dtype
        if x.dtype != lcm_dtype:
            x = x.to(dtype=lcm_dtype)
        return x + self.lcm(x)


def _build_lcm(state_dict, num_channels=16):
    module = LocalContinuityModule(num_channels)
    sd = dict(state_dict)
    if not any(k.startswith("lcm.") for k in sd):
        sd = {f"lcm.{k}": v for k, v in sd.items()}
    module.load_state_dict(sd, strict=False)
    return module


def _load_lcm_weights(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return comfy.utils.load_torch_file(path)


# ---------------------------------------------------------------------------
# Weight Resolution (Prioritizing local T: drive ComfyUI models)
# ---------------------------------------------------------------------------
def _resolve_file(filename):
    for folder in ("geometry_estimation", "lotus2", "loras"):
        try:
            p = folder_paths.get_full_path(folder, filename)
        except Exception:
            p = None
        if p and os.path.exists(p):
            return p

    model_dirs = folder_paths.get_folder_paths("geometry_estimation")
    target_dir = model_dirs[0] if model_dirs else os.path.join(folder_paths.models_dir, "geometry_estimation")
    target_path = os.path.join(target_dir, filename)
    if os.path.exists(target_path):
        return target_path

    os.makedirs(target_dir, exist_ok=True)
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=target_dir)
    except Exception:
        return hf_hub_download(repo_id=ALT_HF_REPO, filename=filename, local_dir=target_dir)


# ---------------------------------------------------------------------------
# Built-in Auto VAE and Empty-Prompt Caches (Zero-wiring support)
# ---------------------------------------------------------------------------
_CACHED_VAE = None
_CACHED_VAE_PATH = None
_CACHED_EMPTY_PROMPT = None


def _get_flux_vae():
    global _CACHED_VAE, _CACHED_VAE_PATH
    vae_path = folder_paths.get_full_path("vae", "ae.safetensors")
    if not vae_path or not os.path.exists(vae_path):
        for f in folder_paths.get_filename_list("vae"):
            if "ae" in f.lower() or "flux" in f.lower():
                p = folder_paths.get_full_path("vae", f)
                if p and os.path.exists(p):
                    vae_path = p
                    break

    if not vae_path or not os.path.exists(vae_path):
        raise FileNotFoundError(
            "FLUX VAE (ae.safetensors) not found in ComfyUI models/vae.\n"
            "Please place ae.safetensors in models/vae/ or wire an external VAE into the node."
        )

    if _CACHED_VAE is not None and _CACHED_VAE_PATH == vae_path:
        return _CACHED_VAE

    _CACHED_VAE = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
    _CACHED_VAE_PATH = vae_path
    return _CACHED_VAE


def _get_flux_empty_prompt():
    global _CACHED_EMPTY_PROMPT
    if _CACHED_EMPTY_PROMPT is not None:
        return _CACHED_EMPTY_PROMPT["cond"], _CACHED_EMPTY_PROMPT["pooled"]

    embed_path = os.path.join(current_dir, "configs", "flux_empty_embed.pt")
    if os.path.exists(embed_path):
        try:
            cached = torch.load(embed_path, weights_only=True)
            _CACHED_EMPTY_PROMPT = cached
            return cached["cond"], cached["pooled"]
        except Exception:
            pass

    raise RuntimeError(
        "Built-in FLUX empty prompt file configs/flux_empty_embed.pt missing.\n"
        "Please connect a conditioning input: DualCLIPLoader -> CLIPTextEncode(\"\")."
    )


# ---------------------------------------------------------------------------
# Scheduling Math
# ---------------------------------------------------------------------------
def _calculate_shift(image_seq_len):
    m = (MAX_SHIFT - BASE_SHIFT) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN)
    b = BASE_SHIFT - m * BASE_IMAGE_SEQ_LEN
    return image_seq_len * m + b


def _sharpener_sigmas(num_steps, image_seq_len):
    sigmas = [1.0 - i * (1.0 - 1.0 / num_steps) / max(num_steps - 1, 1) for i in range(num_steps)]
    if num_steps == 1:
        sigmas = [1.0]
    mu = _calculate_shift(image_seq_len)
    sigmas = [math.exp(mu) / (math.exp(mu) + (1.0 / s - 1.0)) for s in sigmas]
    sigmas.append(0.0)
    return sigmas


# ---------------------------------------------------------------------------
# Colorization (Pure PyTorch LUT)
# ---------------------------------------------------------------------------
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


def _build_lut(n=256):
    anchors = torch.tensor(_SPECTRAL_ANCHORS, dtype=torch.float32)
    xs = torch.linspace(0.0, 1.0, n)
    seg = len(_SPECTRAL_ANCHORS) - 1
    pos = xs * seg
    lo = pos.floor().clamp(0, seg - 1).long()
    frac = (pos - lo).unsqueeze(-1)
    return anchors[lo] * (1 - frac) + anchors[lo + 1] * frac


_SPECTRAL_LUT = _build_lut(256)


def _colorize_depth(depth, colormap="auto"):
    if colormap in ("auto", "gray"):
        norm = []
        for d in depth:
            dmin, dmax = d.min(), d.max()
            norm.append((d - dmin) / (dmax - dmin + 1e-8))
        norm = torch.stack(norm, dim=0)
        return norm.unsqueeze(-1).repeat(1, 1, 1, 3)
    elif colormap == "spectral":
        lut = _SPECTRAL_LUT.to(depth.device)
        out = []
        for d in depth:
            d = d.float()
            dmin, dmax = d.min(), d.max()
            d = (d - dmin) / (dmax - dmin + 1e-8)
            d = 1.0 - d
            idx = (d * 255.0).round().clamp(0, 255).long()
            out.append(lut[idx])
        return torch.stack(out, dim=0)
    elif colormap == "turbo":
        from comfy.ldm.colormap import turbo as _turbo
        norm = []
        for d in depth:
            dmin, dmax = d.min(), d.max()
            norm.append((d - dmin) / (dmax - dmin + 1e-8))
        norm = torch.stack(norm, dim=0)
        return _turbo(norm.clamp(0.0, 1.0))
    else:
        norm = []
        for d in depth:
            dmin, dmax = d.min(), d.max()
            norm.append((d - dmin) / (dmax - dmin + 1e-8))
        norm = torch.stack(norm, dim=0)
        return norm.unsqueeze(-1).repeat(1, 1, 1, 3)


# ---------------------------------------------------------------------------
# Image Resizing Helpers
# ---------------------------------------------------------------------------
def _pick_process_res(max_edge, mode):
    if mode == "native":
        return None
    if mode == "auto":
        if max_edge > 1024:
            return 1024
        if max_edge < 512:
            return 512
        return None
    return int(mode)


def _resize_for_processing(image_bhwc, process_res):
    x = image_bhwc.movedim(-1, 1)
    h, w = x.shape[2], x.shape[3]
    if process_res is not None:
        max_edge = max(h, w)
        if max_edge > process_res:
            scale = process_res / max_edge
            x = F.interpolate(x, size=(max(int(h * scale), 16), max(int(w * scale), 16)), mode="bilinear", align_corners=False)
            h, w = x.shape[2], x.shape[3]

    min_side = min(h, w)
    scale = (min_side // 16) * 16 / min_side
    new_h = max((int(h * scale) // 16) * 16, 16)
    new_w = max((int(w * scale) // 16) * 16, 16)
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return x.movedim(1, -1)


# ---------------------------------------------------------------------------
# Core Inference Runner
# ---------------------------------------------------------------------------
def _run_lotus2_inference(
    model,
    image,
    mode="depth",
    steps=1,
    process_res="auto",
    guidance=3.5,
    colormap="spectral",
    conditioning=None,
    vae=None,
    unload_model=False,
):
    names = FILES[mode]
    core_file = _resolve_file(names["core"])
    sharp_file = _resolve_file(names["sharpener"])
    lcm_file = _resolve_file(names["lcm"])

    core_patcher, _ = comfy.sd.load_lora_for_models(model, None, comfy.utils.load_torch_file(core_file), 1.0, 0.0)
    sharpener_patcher, _ = comfy.sd.load_lora_for_models(model, None, comfy.utils.load_torch_file(sharp_file), 1.0, 0.0)

    lcm_sd = _load_lcm_weights(lcm_file)
    lcm = _build_lcm(lcm_sd)

    if vae is None:
        vae = _get_flux_vae()

    device = mm.get_torch_device()
    dtype = getattr(core_patcher.model, "manual_cast_dtype", None) or core_patcher.model.get_dtype()

    input_h, input_w = image.shape[1], image.shape[2]
    res = _pick_process_res(max(input_h, input_w), process_res)
    proc_image = _resize_for_processing(image[..., :3], res)

    # 1. Obtain Conditioning Tensors (user input OR built-in official empty prompt)
    if conditioning is not None:
        cond = conditioning[0][0].to(device=device, dtype=dtype)
        if len(conditioning[0]) > 1 and isinstance(conditioning[0][1], dict) and "pooled_output" in conditioning[0][1]:
            pooled = conditioning[0][1]["pooled_output"].to(device=device, dtype=dtype)
        else:
            _, cached_pooled = _get_flux_empty_prompt()
            pooled = cached_pooled.to(device=device, dtype=dtype)
    else:
        cached_cond, cached_pooled = _get_flux_empty_prompt()
        cond = cached_cond.to(device=device, dtype=dtype)
        pooled = cached_pooled.to(device=device, dtype=dtype)

    # 2. VAE encode into FLUX latent space
    latent = vae.encode(proc_image)
    batch = latent.shape[0]

    latent = core_patcher.model.process_latent_in(latent).to(device=device, dtype=dtype)
    if cond.shape[0] == 1 and batch > 1:
        cond = cond.repeat(batch, 1, 1)
    if pooled.shape[0] == 1 and batch > 1:
        pooled = pooled.repeat(batch, 1)
    guid = torch.full((batch,), float(guidance), device=device, dtype=dtype)

    pbar = comfy.utils.ProgressBar(1 if steps <= 1 else (1 + steps))

    try:
        with torch.inference_mode():
            # 3. Stage 1: Core Predictor (Deterministic single step at t = 1/1000)
            mm.load_models_gpu([core_patcher])
            core = core_patcher.model.diffusion_model
            t_core = torch.full((batch,), 1.0 / 1000.0, device=device, dtype=dtype)
            coarse = core(latent, t_core, cond, y=pooled, guidance=guid, transformer_options={})
            pbar.update(1)

            # 4. LCM on unpacked latent
            lcm = lcm.to(device=device, dtype=dtype)
            coarse = lcm(coarse)

            # 5. Stage 2: Detail Sharpener (if steps > 1)
            if steps > 1:
                core_patcher.unpatch_model(model.offload_device)
                mm.load_models_gpu([sharpener_patcher])
                sharp = sharpener_patcher.model.diffusion_model

                image_seq_len = (latent.shape[2] // 2) * (latent.shape[3] // 2)
                sigmas = _sharpener_sigmas(steps, image_seq_len)
                lat = coarse
                for i in range(steps):
                    sigma, sigma_next = sigmas[i], sigmas[i + 1]
                    t_val = sigma * NUM_TRAIN_TIMESTEPS / 1000.0
                    t = torch.full((batch,), t_val, device=device, dtype=dtype)
                    v = sharp(lat.to(dtype), t, cond, y=pooled, guidance=guid, transformer_options={})
                    lat = lat + (sigma_next - sigma) * v
                    pbar.update(1)

                out_latent = sharpener_patcher.model.process_latent_out(lat.float())
            else:
                out_latent = core_patcher.model.process_latent_out(coarse.float())

            # 6. VAE decode
            decoded = vae.decode(out_latent)
            decoded = F.interpolate(decoded.permute(0, 3, 1, 2), size=(input_h, input_w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).clamp(0.0, 1.0)
    finally:
        core_patcher.unpatch_model(model.offload_device)
        sharpener_patcher.unpatch_model(model.offload_device)
        if unload_model:
            mm.unload_all_models()
            mm.soft_empty_cache()

    # 7. Post-processing visualization
    if mode == "depth":
        depth = decoded.mean(dim=-1)
        vis = _colorize_depth(depth, colormap=colormap)
    else:  # normal
        vis = decoded

    return (vis.float().cpu(),)


# ===========================================================================
# Node: All-In-One Lotus-2 FLUX Node
# ===========================================================================
class AILab_Lotus2AllInOne:
    """All-in-one Lotus-2 SOTA Geometric Estimation Node. Only image & model required!"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image or video frames."}),
                "model": ("MODEL", {"tooltip": "FLUX.1-dev model (e.g. flux1-dev-fp8.safetensors or GGUF)."}),
            },
            "optional": {
                "mode": (["depth", "normal"], {"default": "depth", "tooltip": "Geometry prediction mode: depth (Disparity Depth) or normal (Surface Normal vector map)."}),
                "steps": ("INT", {"default": 1, "min": 1, "max": 10, "tooltip": "1 = Instant 1-step Core Predictor (~15s); 2-10 = Multi-step detail sharpener."}),
                "resolution": (["auto", "768", "512", "1024", "native"], {"default": "auto", "tooltip": "Processing resolution."}),
                "colormap": (["auto", "gray", "spectral", "turbo"], {"default": "auto", "tooltip": "auto: Standard Normal for normal mode, Clean Grayscale for depth mode. Or pick spectral/turbo."}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "FLUX guidance scale."}),
                "conditioning": ("CONDITIONING", {"tooltip": "Optional. If not connected, automatically uses built-in authentic FLUX empty prompt."}),
                "vae": ("VAE", {"tooltip": "Optional. If not connected, automatically loads local ae.safetensors."}),
                "unload_model": ("BOOLEAN", {"default": False, "tooltip": "Fully unload models from VRAM after execution."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "🧪AILab/Geometry"
    TITLE = "Lotus-2 (FLUX)"

    def process(
        self,
        image,
        model,
        mode="depth",
        steps=1,
        resolution="auto",
        colormap="auto",
        guidance=3.5,
        conditioning=None,
        vae=None,
        unload_model=False,
    ):
        return _run_lotus2_inference(
            model=model,
            image=image,
            mode=mode,
            steps=steps,
            process_res=resolution,
            guidance=guidance,
            colormap=colormap,
            conditioning=conditioning,
            vae=vae,
            unload_model=unload_model,
        )


NODE_CLASS_MAPPINGS = {
    "Lotus2": AILab_Lotus2AllInOne,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Lotus2": "Lotus-2 (FLUX)",
}


