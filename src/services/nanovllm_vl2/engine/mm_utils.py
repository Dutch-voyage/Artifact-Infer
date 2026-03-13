import base64
import hashlib
import tempfile
import numpy as np
import torch
from PIL import Image
import requests
import pybase64
from io import BytesIO
from typing import Union
import triton.language as tl
import triton
import os

def load_image(
    image_file: Union[Image.Image, str, bytes],
) -> tuple[Image.Image, tuple[int, int]]:

    image = image_size = None
    if isinstance(image_file, Image.Image):
        image = image_file
        image_size = (image.width, image.height)
    elif isinstance(image_file, bytes):
        image = Image.open(BytesIO(image_file))
    elif image_file.startswith("http://") or image_file.startswith("https://"):
        timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
        response = requests.get(image_file, stream=True, timeout=timeout)
        try:
            response.raise_for_status()
            image = Image.open(response.raw)
            image.load()  # Force loading to avoid issues after closing the stream
        finally:
            response.close()
    elif image_file.lower().endswith(("png", "jpg", "jpeg", "webp", "gif")):
        image = Image.open(image_file)
        print(f"[DEBUG]load {image} successfully from path.")
    elif image_file.startswith("data:"):
        image_file = image_file.split(",")[1]
        image = Image.open(BytesIO(pybase64.b64decode(image_file, validate=True)))
    elif isinstance(image_file, str):
        image = Image.open(BytesIO(pybase64.b64decode(image_file, validate=True)))
    else:
        raise ValueError(f"Invalid image: {image_file}")

    return image, image_size

def load_video(video_file: Union[str, bytes], use_gpu: bool = True):
    # We import decord here to avoid a strange Segmentation fault (core dumped) issue.
    from decord import VideoReader, cpu, gpu

    try:
        from decord.bridge import decord_bridge

        ctx = gpu(0)
        _ = decord_bridge.get_ctx_device(ctx)
    except Exception:
        ctx = cpu(0)

    tmp_file = None
    vr = None
    try:
        if isinstance(video_file, bytes):
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp_file.write(video_file)
            tmp_file.close()
            vr = VideoReader(tmp_file.name, ctx=ctx)
        elif isinstance(video_file, str):
            if video_file.startswith(("http://", "https://")):
                timeout = int(os.getenv("REQUEST_TIMEOUT", "10"))
                response = requests.get(video_file, stream=True, timeout=timeout)
                response.raise_for_status()
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file.close()
                vr = VideoReader(tmp_file.name, ctx=ctx)
            elif video_file.startswith("data:"):
                _, encoded = video_file.split(",", 1)
                video_bytes = base64.b64decode(encoded)
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_file.write(video_bytes)
                tmp_file.close()
                vr = VideoReader(tmp_file.name, ctx=ctx)
            elif os.path.isfile(video_file):
                vr = VideoReader(video_file, ctx=ctx)
            else:
                video_bytes = base64.b64decode(video_file)
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_file.write(video_bytes)
                tmp_file.close()
                vr = VideoReader(tmp_file.name, ctx=ctx)
        else:
            raise ValueError(f"Unsupported video input type: {type(video_file)}")

        return vr

    finally:
        if tmp_file and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)

def flatten_nested_list(nested_list):
    if isinstance(nested_list, list):
        return [
            item for sublist in nested_list for item in flatten_nested_list(sublist)
        ]
    else:
        return [nested_list]


FMIX32_C1 = 0x85EBCA6B
FMIX32_C2 = 0xC2B2AE35
POS_C1 = 0x27D4EB2D
POS_C2 = 0x165667B1

@triton.jit
def _rotl32(x, r: tl.constexpr):
    return (x << r) | (x >> (32 - r))


@triton.jit
def _fmix32(x, C1: tl.constexpr, C2: tl.constexpr):
    c1 = tl.full((), C1, tl.uint32)
    c2 = tl.full((), C2, tl.uint32)
    x ^= x >> 16
    x = x * c1
    x ^= x >> 13
    x = x * c2
    x ^= x >> 16
    return x

@triton.jit
def hash_tiles32_kernel_blocked(
    in_ptr,
    out_ptr,
    n_u32,
    seed1,
    seed2,
    FM_C1: tl.constexpr,
    FM_C2: tl.constexpr,
    POS_A: tl.constexpr,
    POS_B: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_CG: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base = pid * TILE

    s1 = tl.full((), seed1, tl.uint32)
    s2 = tl.full((), seed2, tl.uint32)
    posA = tl.full((), POS_A, tl.uint32)
    posB = tl.full((), POS_B, tl.uint32)

    h1 = tl.zeros((), dtype=tl.uint32)
    h2 = tl.zeros((), dtype=tl.uint32)

    for off in tl.static_range(0, TILE, BLOCK):
        idx = base + off + tl.arange(0, BLOCK)
        m = idx < n_u32

        if USE_CG:
            v = tl.load(in_ptr + idx, mask=m, other=0, cache_modifier=".cg")
        else:
            v = tl.load(in_ptr + idx, mask=m, other=0)
        v = v.to(tl.uint32)

        iu = idx.to(tl.uint32)
        p1 = (iu * posA + s1) ^ _rotl32(iu, 15)
        p2 = (iu * posB + s2) ^ _rotl32(iu, 13)

        k1 = _fmix32(v ^ p1, C1=FM_C1, C2=FM_C2)
        k2 = _fmix32(v ^ p2, C1=FM_C1, C2=FM_C2)

        zero32 = tl.zeros_like(k1)
        k1 = tl.where(m, k1, zero32)
        k2 = tl.where(m, k2, zero32)

        h1 += tl.sum(k1, axis=0).to(tl.uint32)
        h2 += tl.sum(k2, axis=0).to(tl.uint32)

    nbytes = tl.full((), n_u32 * 4, tl.uint32)
    h1 ^= nbytes
    h2 ^= nbytes
    h1 = _fmix32(h1, C1=FM_C1, C2=FM_C2)
    h2 = (
        _fmix32(h2, C1=FMIX32_C1, C2=FMIX32_C2)
        if False
        else _fmix32(h2, C1=FM_C1, C2=FM_C2)
    )

    out = (h1.to(tl.uint64) << 32) | h2.to(tl.uint64)
    tl.store(out_ptr + pid, out)


@triton.jit
def add_tree_reduce_u64_kernel(in_ptr, out_ptr, n_elems, CHUNK: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * CHUNK
    h = tl.zeros((), dtype=tl.uint64)
    for i in tl.static_range(0, CHUNK):
        idx = start + i
        m = idx < n_elems
        v = tl.load(in_ptr + idx, mask=m, other=0).to(tl.uint64)
        h += v
    tl.store(out_ptr + pid, h)


def _as_uint32_words(t: torch.Tensor) -> torch.Tensor:
    assert t.is_cuda, "Use .cuda() first"
    tb = t.contiguous().view(torch.uint8)
    nbytes = tb.numel()
    pad = (4 - (nbytes & 3)) & 3
    if pad:
        tb_p = torch.empty(nbytes + pad, dtype=torch.uint8, device=tb.device)
        tb_p[:nbytes].copy_(tb)
        tb_p[nbytes:].zero_()
        tb = tb_p
    return tb.view(torch.uint32)


def _final_splitmix64(x: int) -> int:
    mask = (1 << 64) - 1
    x &= mask
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & mask
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & mask
    x ^= x >> 31
    return x

@torch.inference_mode()
def gpu_tensor_hash(
    tensor: torch.Tensor,
    *,
    seed: int = 0x243F6A88,
    tile_words: int = 8192,
    block_words: int = 256,
    reduce_chunk: int = 1024,
    num_warps: int = 4,
    num_stages: int = 4,
    use_cg: bool = True,
) -> int:
    assert tensor.is_cuda, "Use .cuda() first"
    u32 = _as_uint32_words(tensor)
    n = u32.numel()
    if n == 0:
        return 0

    grid1 = (triton.cdiv(n, tile_words),)
    partials = torch.empty(grid1[0], dtype=torch.uint64, device=u32.device)
    hash_tiles32_kernel_blocked[grid1](
        u32,
        partials,
        n,
        seed1=seed & 0xFFFFFFFF,
        seed2=((seed * 0x9E3779B1) ^ 0xDEADBEEF) & 0xFFFFFFFF,
        FM_C1=FMIX32_C1,
        FM_C2=FMIX32_C2,
        POS_A=POS_C1,
        POS_B=POS_C2,
        TILE=tile_words,
        BLOCK=block_words,
        USE_CG=use_cg,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    cur = partials
    while cur.numel() > 1:
        n_elems = cur.numel()
        grid2 = (triton.cdiv(n_elems, reduce_chunk),)
        nxt = torch.empty(grid2[0], dtype=torch.uint64, device=cur.device)
        add_tree_reduce_u64_kernel[grid2](cur, nxt, n_elems, CHUNK=reduce_chunk)
        cur = nxt

    return _final_splitmix64(int(cur.item()))


def tensor_hash(tensor_list) -> int:
    """
    hash a tensor or a tensor list
    """
    tensor = tensor_list
    if isinstance(tensor_list, list):
        tensor_list = flatten_nested_list(tensor_list)
        tensor_list = [
            x.flatten() if isinstance(x, torch.Tensor) else x for x in tensor_list
        ]
        tensor = torch.concat(tensor_list)
    if tensor.is_cuda:
        return gpu_tensor_hash(tensor.cuda())
    tensor = tensor.detach().contiguous()

    if tensor.dtype == torch.bfloat16:
        # memoryview() doesn't support PyTorch's BFloat16 dtype
        tensor = tensor.float()

    assert isinstance(tensor, torch.Tensor)
    tensor_cpu = tensor.cpu()

    mv = memoryview(tensor_cpu.numpy())
    return data_hash(mv.tobytes())


def data_hash(data) -> int:
    hash_bytes = hashlib.sha256(data).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big", signed=False)

def hash_feature(f):
    if isinstance(f, list):
        if isinstance(f[0], torch.Tensor):
            return tensor_hash(f)
        return data_hash(tuple(flatten_nested_list(f)))
    elif isinstance(f, np.ndarray):
        arr = np.ascontiguousarray(f)
        arr_bytes = arr.tobytes()
        return data_hash(arr_bytes)
    elif isinstance(f, torch.Tensor):
        return tensor_hash([f])
    return data_hash(f)