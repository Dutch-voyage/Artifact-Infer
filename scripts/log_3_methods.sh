# python -m eval.test_van --compress_method none --attn_reduce_method raw --lse_preserve_merge False
python -m eval.test_van --compress_method none --attn_reduce_method raw --lse_preserve_merge False --if_fake_compress True --p_attn 0.60
python -m eval.test_van --compress_method none --attn_reduce_method raw --lse_preserve_merge False --if_fake_compress True --p_attn 0.60

# python -m eval.test_van --compress_method none --attn_reduce_method raw --lse_preserve_merge True
# python -m eval.test_van --compress_method snapkv --attn_reduce_method maxpool_merge --lse_preserve_merge True
# python -m eval.test_van --compress_method rkv --attn_reduce_method sim_merge --lse_preserve_merge True