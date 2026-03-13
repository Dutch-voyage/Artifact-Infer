import torch
from src.artifacts.nanovllm_v6.layers.rotary_embedding import RotaryEmbedding

def test_inverse_rotary():
    head_size = 128
    rotary_dim = 128
    max_position_embeddings = 1024
    base = 10000.0

    rotary_embedding = RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
    ).to("cuda")

    num_kv_heads = 8
    num_q_heads = 32
    
    positions = torch.arange(0, 512).to("cuda")
    query = torch.randn(512, num_kv_heads, head_size).to("cuda")
    key = torch.randn(512, num_kv_heads, head_size).to("cuda")

    # Apply rotary embedding
    rotated_query, rotated_key = rotary_embedding(positions, query, key)

    # Inverse rotary embedding on the key
    recovered_key = rotary_embedding.inverse(positions, rotated_key)

    # Check if the recovered key is close to the original key
    assert torch.allclose(recovered_key, key, atol=1e-5), "Inverse rotary embedding failed"
    
    print("Inverse rotary embedding test passed!")
    
if __name__ == "__main__":
    test_inverse_rotary()