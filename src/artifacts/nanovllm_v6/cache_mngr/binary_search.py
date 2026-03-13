import triton
import triton.language as tl
import torch
import torch.nn.functional as F
from functools import partial
from flashinfer.sampling import top_p_renorm_probs
from .utils import gather_num_selected


@triton.jit
def reduce_num_selected_kernel(
    logits_ptr,
    num_selected_ptr,
    reduce_ptr, 
    locks_ptr, 
    N: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BN)
    pid_n = pid // num_pid_n
    pid_bn = pid % num_pid_n
    
    num_selected = tl.load(num_selected_ptr + pid_n)
    offsets = pid_bn * BN + tl.arange(0, BN)
    logits_selected = tl.load(logits_ptr + pid_n * N + offsets, mask=offsets < num_selected, other=0.0)
    total = tl.sum(logits_selected, axis=0)
    
    this_lock = locks_ptr + pid_n
    while tl.atomic_cas(this_lock, 0, 1) == 1: 
        pass
    
    reduce = tl.load(reduce_ptr + pid_n)
    reduce += total
    tl.store(reduce_ptr + pid_n, reduce)
    
    tl.debug_barrier()
    tl.atomic_xchg(this_lock, 0)

def reduce_num_selected(
    logits, 
    num_selected, 
):
    # assert torch.all(num_selected <= logits.shape[-1])
    bsz, kv_len = logits.shape
    reduce = torch.zeros((bsz,), dtype=logits.dtype, device=logits.device)
    BN = 32
    
    locks = torch.full((logits.shape[0],), 0, dtype=torch.uint32).to(logits.device)
    
    reduce_num_selected_kernel[(bsz * triton.cdiv(kv_len, BN) ,)](
        logits,
        num_selected,
        reduce,
        locks, 
        kv_len, 
        BN, 
    )
    
    return reduce

def calculate_mass(shifted_logits, T, num_selected, transform_func):
    scaled_logits = shifted_logits / T.unsqueeze(-1)
    scaled_logits = transform_func(scaled_logits)
    scaled_logits -= scaled_logits.amax(dim=-1, keepdim=True)
    probs = F.softmax(scaled_logits, dim=-1)
    mass = reduce_num_selected(probs, num_selected)
    return mass

def calculate_mass_differentiable(shifted_logits, T, num_selected, transform_func):
    """Differentiable version of calculate_mass for gradient descent."""

    T = T.unsqueeze(-1).repeat(1, shifted_logits.shape[0] // T.shape[0]).reshape(-1)
    scaled_logits = shifted_logits / T.unsqueeze(-1)
    scaled_logits = transform_func(scaled_logits)
    scaled_logits -= scaled_logits.amax(dim=-1, keepdim=True).detach()
    probs = F.softmax(scaled_logits, dim=-1)

    # Create a mask for selected elements (differentiable)
    bsz, kv_len = probs.shape
    indices = torch.arange(kv_len, device=probs.device).unsqueeze(0).expand(bsz, -1)
    mask = (indices < num_selected.unsqueeze(-1)).float()

    # Calculate mass using standard PyTorch operations (preserves gradients)
    mass = (probs * mask).sum(dim=-1)
    return mass

def calculate_mass_differentiable_linear(shifted_probs, T, num_selected):
    """Differentiable version of calculate_mass for gradient descent."""
    shifted_probs -= shifted_probs.amin(dim=-1, keepdim=True).detach()
    shifted_probs = shifted_probs / shifted_probs.sum(dim=-1, keepdim=True)
    logits = torch.log(shifted_probs + 1e-10)
    logits = logits / T.unsqueeze(-1)
    logits -= logits.amax(dim=-1, keepdim=True).detach()
    probs = F.softmax(logits, dim=-1)

    # Create a mask for selected elements (differentiable)
    bsz, kv_len = probs.shape
    indices = torch.arange(kv_len, device=probs.device).unsqueeze(0).expand(bsz, -1)
    mask = (indices < num_selected.unsqueeze(-1)).float()

    # Calculate mass using standard PyTorch operations (preserves gradients)
    mass = (probs * mask).sum(dim=-1)
    return mass

def calculate_mass_linear(shifted_probs, T, num_selected):
    logits = torch.log(shifted_probs + 1e-10)
    logits = logits / T.unsqueeze(-1)
    
    logits -= logits.amax(dim=-1, keepdim=True)
    scaled_probs = torch.exp(logits)

    scaled_probs = scaled_probs / scaled_probs.sum(dim=-1, keepdim=True)
    mass = reduce_num_selected(scaled_probs, num_selected)
    return mass

def binary_search_T(raw_logits, anchor_p, transform_func):
    bsz, num_heads, q_cache_len, kv_cache_len = raw_logits.shape
    # bsz, num_heads, kv_cache_len = raw_logits.shape
    raw_logits = raw_logits.reshape(-1, kv_cache_len)
    T_min = torch.ones(raw_logits.shape[0], device="cuda", dtype=torch.float64) * 1e-3
    T_max = torch.ones(raw_logits.shape[0], device="cuda", dtype=torch.float64) * 1
    normed_probs = top_p_renorm_probs(F.softmax(raw_logits, dim=-1), top_p=anchor_p)
    selected_indices = torch.vmap(partial(torch.nonzero_static, size=normed_probs.shape[-1]), in_dims=(0,))(normed_probs).squeeze(-1)
    num_selected_oracle = gather_num_selected(
        selected_indices 
    )
    
    T_mid = (T_min + T_max) / 2.0
    cu_mass = calculate_mass(raw_logits, T_mid, num_selected_oracle, transform_func)
    while (torch.all(T_min < T_max) and torch.abs(cu_mass - anchor_p).max() > 1e-2):
        T_min = torch.where(cu_mass > anchor_p, T_mid, T_min)
        T_max = torch.where(cu_mass <= anchor_p, T_mid, T_max)
        T_mid = (T_min + T_max) / 2.0
        cu_mass = calculate_mass(raw_logits, T_mid, num_selected_oracle, transform_func)
    # print(cu_mass.min().item(), cu_mass.max().item(), cu_mass.mean().item())
    # print("-" * 100)
    T_mid = T_mid.reshape(bsz, num_heads, q_cache_len)
    raw_logits = raw_logits.reshape(bsz, num_heads, q_cache_len, kv_cache_len)
    raw_logits = raw_logits / T_mid.unsqueeze(-1)
    transformed_logits = transform_func(raw_logits.view(-1, kv_cache_len)).view(bsz, num_heads, q_cache_len, kv_cache_len)
    
    # T_mid = T_mid.reshape(bsz, num_heads)
    # raw_logits = raw_logits.reshape(bsz, num_heads, kv_cache_len)
    # raw_logits = raw_logits / T_mid.unsqueeze(-1)
    # transformed_logits = transform_func(raw_logits.view(-1, kv_cache_len)).view(bsz, num_heads, kv_cache_len)
    
    return transformed_logits, T_mid

def gradient_descent_T(raw_logits, anchor_p, transform_func, lr=1e-2, max_iters=50):
    bsz, num_heads, q_cache_len, kv_cache_len = raw_logits.shape
    # bsz, num_heads, kv_cache_len = raw_logits.shape
    raw_logits = raw_logits.reshape(-1, kv_cache_len)
    T = torch.full((bsz * num_heads,), 0.5, device="cuda", dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([T], lr=lr)
    # optimizer = torch.optim.SGD([T], lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    normed_probs = top_p_renorm_probs(F.softmax(raw_logits, dim=-1), top_p=anchor_p)
    selected_indices = torch.vmap(partial(torch.nonzero_static, size=normed_probs.shape[-1]), in_dims=(0,))(normed_probs).squeeze(-1)
    num_selected_oracle = gather_num_selected(
        selected_indices
    )

    for iter in range(max_iters):
        # Use differentiable version for gradient-based optimization
        cu_mass = calculate_mass_differentiable(raw_logits, T, num_selected_oracle, transform_func)
        loss = torch.mean((cu_mass - anchor_p) ** 2) ** 0.5
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Ensure T stays positive and reasonable
        with torch.no_grad():
            T.clamp_(min=1e-3, max=10.0)
        if torch.abs(cu_mass - anchor_p).max() < 1e-2:
            break
    # print(cu_mass.max().item(), cu_mass.min().item(), cu_mass.mean().item())
    # print("-" * 100)
    T = T.detach().reshape(bsz, num_heads)
    raw_logits = raw_logits.reshape(bsz, num_heads, q_cache_len, kv_cache_len)
    raw_logits = raw_logits / T.unsqueeze(-1).unsqueeze(-1)
    transformed_logits = transform_func(raw_logits.view(-1, kv_cache_len)).view(bsz, num_heads, q_cache_len, kv_cache_len)
    
    # T = T.detach().reshape(bsz, num_heads)
    # raw_logits = raw_logits.reshape(bsz, num_heads, kv_cache_len)
    # raw_logits = raw_logits / T.unsqueeze(-1)
    # transformed_logits = transform_func(raw_logits.view(-1, kv_cache_len)).view(bsz, num_heads, kv_cache_len)
    
    return transformed_logits, T

def gradient_descent_T_linear(raw_probs, shifted_probs, anchor_p, lr=1e-2, max_iters=50):
    # bsz, num_heads, q_cache_len, kv_cache_len = raw_probs.shape
    bsz, num_heads, kv_cache_len = shifted_probs.shape
    shifted_probs = shifted_probs.reshape(-1, kv_cache_len)
    raw_probs = raw_probs.reshape(-1, kv_cache_len)
    T = torch.full((raw_probs.shape[0],), 0.5, device="cuda", dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([T], lr=lr)

    normed_probs = top_p_renorm_probs(raw_probs, top_p=anchor_p)
    selected_indices = torch.vmap(partial(torch.nonzero_static, size=normed_probs.shape[-1]), in_dims=(0,))(normed_probs).squeeze(-1)
    num_selected_oracle = gather_num_selected(
        selected_indices
    )

    for iter in range(max_iters):
        # Use differentiable version for gradient-based optimization
        cu_mass = calculate_mass_differentiable_linear(shifted_probs, T, num_selected_oracle)
        loss = torch.mean((cu_mass - anchor_p) ** 2)
        # loss = torch.mean(torch.abs(cu_mass - anchor_p))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Ensure T stays positive and reasonable
        with torch.no_grad():
            T.clamp_(min=1e-3, max=10.0)
        if torch.abs(cu_mass - anchor_p).max() < 1e-2:
            break
    # print(cu_mass.max().item(), cu_mass.min().item(), cu_mass.mean().item())
    # print(T.max().item(), T.min().item())
    # print("-" * 100)
    # T = T.detach().reshape(bsz, num_heads, q_cache_len)
    T = T.detach().reshape(bsz, num_heads)
    # shifted_probs = shifted_probs.reshape(bsz, num_heads, q_cache_len, kv_cache_len)
    shifted_probs = shifted_probs.reshape(bsz, num_heads, kv_cache_len)
    shifted_probs = torch.softmax(torch.log(shifted_probs + 1e-10) / T.unsqueeze(-1), dim=-1)
    return shifted_probs, T


def binary_search_T_linear(raw_probs, shifted_probs, anchor_p = 0.6):
    bsz, num_heads, q_cache_len, kv_cache_len = shifted_probs.shape
    shifted_probs = shifted_probs.reshape(-1, kv_cache_len).to(torch.float32)
    # shifted_probs = shifted_probs / shifted_probs.sum(dim=-1, keepdim=True)    
    raw_probs = raw_probs.reshape(-1, kv_cache_len)
    T_min = torch.ones(shifted_probs.shape[0], device="cuda") * 1e-3
    T_max = torch.ones(shifted_probs.shape[0], device="cuda") * 1
    
    normed_probs = top_p_renorm_probs(raw_probs, top_p=anchor_p)
    
    selected_indices = torch.vmap(partial(torch.nonzero_static, size=normed_probs.shape[-1]), in_dims=(0,))(normed_probs).squeeze(-1)
    num_selected_oracle = gather_num_selected(
        selected_indices 
    )
    
    T_mid = (T_min + T_max) / 2.0
    cu_mass = calculate_mass_linear(shifted_probs, T_mid, num_selected_oracle)
    while (torch.all(T_min < T_max) and torch.abs(cu_mass - anchor_p).max() > 1e-2):
        T_min = torch.where(cu_mass > anchor_p, T_mid, T_min)
        T_max = torch.where(cu_mass <= anchor_p, T_mid, T_max)
        T_mid = (T_min + T_max) / 2.0
        cu_mass = calculate_mass_linear(shifted_probs, T_mid, num_selected_oracle)
    print(cu_mass.min().item(), cu_mass.max().item())
    print(T_mid.min().item(), T_mid.max().item())
    print("-" * 100)

    T_mid = T_mid.reshape(bsz, num_heads, q_cache_len)
    
    shifted_probs = shifted_probs.reshape(bsz, num_heads, q_cache_len, kv_cache_len)
    shifted_probs = torch.exp((torch.log(shifted_probs + 1e-10) / T_mid.unsqueeze(-1)))
    shifted_probs = shifted_probs / shifted_probs.sum(dim=-1, keepdim=True)
    return shifted_probs, T_mid
