uv run python -m eval.test_bazaar_moe \
    --world-size 8 \
    --tp-size 1 \
    --moe-structure ep_ll_triton \
    --max-model-len 256 \
    --max-num-batched-tokens 256 