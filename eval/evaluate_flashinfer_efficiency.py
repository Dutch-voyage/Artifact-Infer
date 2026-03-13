import torch
import flashinfer
import time

# qwen3-4B
num_layers = 36
num_qo_heads = 32
num_kv_heads = 8
head_dim = 128

max_num_pages = 512
page_size = 1
# allocate 128MB workspace buffer
workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    workspace_buffer, "NHD"
)
batch_size = 1
kv_page_indices = torch.arange(max_num_pages).int().to("cuda:0")

kv_page_indptr = torch.tensor([0, 512], dtype=torch.int32, device="cuda:0")
# 1 <= kv_last_page_len <= page_size
kv_last_page_len = torch.tensor(
    [1], dtype=torch.int32, device="cuda:0"
)
kv_cache_at_layer = [
    torch.randn(
        max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ) for _ in range(num_layers)
]


# warmup
for _ in range(2):
    # create auxiliary data structures for batch decode attention
    decode_wrapper.plan(
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type=torch.float16
    )
    outputs = []

    for i in range(num_layers):
        q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
        kv_cache = kv_cache_at_layer[i]
        # compute batch decode attention, reuse auxiliary data structures for all layers
        o = decode_wrapper.run(q, kv_cache)
        outputs.append(o)


torch.cuda.synchronize()
start_time = time.time()

for _ in range(512):
    # create auxiliary data structures for batch decode attention
    decode_wrapper.plan(
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type=torch.float16
    )
    outputs = []

    for i in range(num_layers):
        q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
        kv_cache = kv_cache_at_layer[i]
        # compute batch decode attention, reuse auxiliary data structures for all layers
        o = decode_wrapper.run(q, kv_cache)
        outputs.append(o)

torch.cuda.synchronize()
end_time = time.time()

print(f"decoding time {512 / (end_time - start_time)} decoding token per second")
