import torch
import torch.nn.functional as F
import numpy as np
import torch
from PIL import Image


import difflib
import diffusers
import numpy as np
import requests
import torch
import torch.nn.functional as F
import transformers
from diffusers import (AutoencoderKL, DiffusionPipeline,
                       FlowMatchEulerDiscreteScheduler, FluxPipeline,
                       FluxTransformer2DModel, SD3Transformer2DModel,
                       StableDiffusion3Pipeline)
from diffusers.callbacks import PipelineCallback
from torchvision import transforms
from transformers import (AutoModelForCausalLM, AutoProcessor, CLIPTextModel,
                          CLIPTextModelWithProjection, T5EncoderModel)


def get_bbox_from_color(mask_array, target_color, tolerance=10):
    r, g, b = target_color
    color_mask = ((np.abs(mask_array[:, :, 0] - r) <= tolerance) &
                  (np.abs(mask_array[:, :, 1] - g) <= tolerance) &
                  (np.abs(mask_array[:, :, 2] - b) <= tolerance))
    coords = np.column_stack(np.where(color_mask))
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def get_rgb_color_masks_1d(
        mask_path: str,
        grid_h: int = 64,
        grid_w: int = 64,
        device: str | torch.device = "cuda",
        tol: int = 20, ):
    """
    Returns:
      red_mask_1d, blue_mask_1d, green_mask_1d  (torch.bool, shape [grid_h*grid_w])
    Order:
      row-major flatten: idx = h * grid_w + w
      (this matches latents reshaped as (H//2)*(W//2) with W fastest)
    """

    # 1) load + resize with NEAREST (라벨 보존)
    mask_img = Image.open(mask_path).convert("RGB")
    mask_img = mask_img.resize((grid_w, grid_h), resample=Image.NEAREST) # 64,64 크기

    # 2) to tensor [H, W, 3] in 0..255
    mask_np = np.array(mask_img, dtype=np.uint8)  # [H,W,3]
    r = mask_np[..., 0].astype(np.int16)
    g = mask_np[..., 1].astype(np.int16)
    b = mask_np[..., 2].astype(np.int16)

    red2d_64_64 = ((r == 255) & (g == 0) & (b == 0)).astype(np.uint8)
    green2d_64_64 = ((r == 0) & (g == 255) & (b == 0)).astype(np.uint8)
    blue2d_64_64 = ((r == 0) & (g == 0) & (b == 255)).astype(np.uint8)

    red_2d   = (abs(r - 255) <= tol) & (g <= tol) & (b <= tol)
    blue_2d  = (r <= tol) & (g <= tol) & (abs(b - 255) <= tol)
    green_2d = (r <= tol) & (abs(g - 255) <= tol) & (b <= tol)

    # 4) flatten to 1D (row-major: W fastest)
    red_1d = torch.from_numpy(red_2d.reshape(-1)).to(device=device)
    blue_1d  = torch.from_numpy(blue_2d.reshape(-1)).to(device=device)
    green_1d = torch.from_numpy(green_2d.reshape(-1)).to(device=device)

    red_1d = red_1d.bool()
    blue_1d = blue_1d.bool()
    green_1d = green_1d.bool()

    return red_1d, blue_1d, green_1d,red2d_64_64, green2d_64_64, blue2d_64_64

def gaussian_kernel2d(k: int, sigma: float, device="cuda", dtype=torch.float32):
    ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    ker = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    ker = ker / ker.sum()
    return ker  # [k,k]

@torch.no_grad()
def soften_onehot_masks(
        red_1d_bool: torch.Tensor,
        blue_1d_bool: torch.Tensor,
        green_1d_bool: torch.Tensor,
        grid_h: int = 64,
        grid_w: int = 64,
        k: int = 9,
        sigma: float = 2.0,
        eps: float = 1e-6,
):
    """
    input: bool [4096]
    output: float [4096] in [0,1], and r+g+b==1 per pixel
    """
    device = red_1d_bool.device
    dtype = torch.float32

    def _blur(mask_1d_bool):
        m = mask_1d_bool.view(1, 1, grid_h, grid_w).to(dtype)
        ker = gaussian_kernel2d(k, sigma, device=device, dtype=dtype).view(1, 1, k, k)
        m = F.conv2d(m, ker, padding=k // 2)
        return m  # [1,1,H,W]

    r = _blur(red_1d_bool)
    g = _blur(green_1d_bool)
    b = _blur(blue_1d_bool)

    s = (r + g + b).clamp_min(eps)
    r = (r / s).clamp(0, 1)
    g = (g / s).clamp(0, 1)
    b = (b / s).clamp(0, 1)

    # flatten back
    return r.view(-1), b.view(-1), g.view(-1)


def get_flux_pipeline(
        model_id="black-forest-labs/FLUX.1-dev",
        pipeline_class=FluxPipeline,
        torch_dtype=torch.bfloat16,
        quantize=False
):
    ############ Diffusion Transformer ############
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch_dtype
    )

    ############ Text Encoder ############
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch_dtype
    )

    ############ Text Encoder 2 ############
    text_encoder_2 = T5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=torch_dtype
    )

    ############ VAE ############
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch_dtype
    )

    pipe = pipeline_class.from_pretrained(model_id,
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch_dtype
    )
    return pipe


def mask_decode(encoded_mask, image_shape=[512, 512]):
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))

    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i] + j] = 1

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1

    return mask_array


def mask_interpolate(mask, size=128):
    mask = torch.tensor(mask)
    mask = F.interpolate(mask[None, None, ...], size, mode='bicubic')
    mask = mask.squeeze()
    return mask


def get_blend_word_index(prompt, word, tokenizer):
    input_ids = tokenizer(prompt).input_ids
    blend_ids = tokenizer(word, add_special_tokens=False).input_ids

    index = []
    for i, id in enumerate(input_ids):
        # Ignore common token
        if id < 100:
            continue
        if id in blend_ids:
            index.append(i)

    return index


def find_token_id_differences(prompt1, prompt2, tokenizer):
    # Tokenize inputs and get input IDs
    tokens1 = tokenizer.encode(prompt1, add_special_tokens=False)
    tokens2 = tokenizer.encode(prompt2, add_special_tokens=False)

    # Get sequence matcher output
    seq_matcher = difflib.SequenceMatcher(None, tokens1, tokens2)

    diff1_indices, diff1_ids = [], []
    diff2_indices, diff2_ids = [], []

    for opcode, a_start, a_end, b_start, b_end in seq_matcher.get_opcodes():
        if opcode in ['replace', 'delete']:
            diff1_indices.extend(range(a_start, a_end))
            diff1_ids.extend(tokens1[a_start:a_end])
        if opcode in ['replace', 'insert']:
            diff2_indices.extend(range(b_start, b_end))
            diff2_ids.extend(tokens2[b_start:b_end])

    return {
        'prompt_1': {'index': diff1_indices, 'id': diff1_ids},
        'prompt_2': {'index': diff2_indices, 'id': diff2_ids}
    }

"""
def find_word_token_indices(prompt, word, tokenizer):
    # Tokenize with offsets to track word positions
    print(f'concept_nouns = {word} with prompt = {prompt}')
    # 여기서 나는 word 가 ['cat' 'roof' 인데??
    # prompt = 는 cat on the  roof at night 이고
    # word = ['cat','roof'] 야
    # 그러면 최소 2개가 나와얄거 같은데,,
    encoding = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    tokens = encoding.tokens()
    word_indices = []
    word_tokens = tokenizer(word, add_special_tokens=False).tokens()
    print(f'word_tokens = {word_tokens}')

    for i in range(len(tokens) - len(word_tokens) + 1):
        if tokens[i: i + len(word_tokens)] == word_tokens:
            word_indices.extend(range(i, i + len(word_tokens)))
    print(f'word_indices = {word_indices}')
    return word_indices
"""
def find_word_token_indices(prompt, word, tokenizer):
    # prompt tokenization
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    tokens = enc.tokens()

    def _find_single(w: str):
        if not isinstance(w, str) or len(w) == 0:
            return []
        w_tokens = tokenizer(w, add_special_tokens=False).tokens()
        hits = []
        for i in range(len(tokens) - len(w_tokens) + 1):
            if tokens[i:i + len(w_tokens)] == w_tokens:
                hits.extend(range(i, i + len(w_tokens)))
        return hits

    if isinstance(word, (list, tuple)):
        all_idx = []
        for w in word:
            all_idx.extend(_find_single(w))
        all_idx = sorted(set(all_idx))
        # 0,3
        return all_idx

    hits = _find_single(word)
    return hits
