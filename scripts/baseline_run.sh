for p_attn in 0.90 0.80 0.70 0.60; do
  for layer_budget in 512 1024 2048; do
      for data_source in aime24 aime25 umathtop50; do
        python -m eval.test_aime_evict \
            --data_source ${data_source} \
            --enforce_eager False \
            --compress_method rkv \
            --if_fake_compress False \
            --layer_budget ${layer_budget} \
            --window_size 32 \
            --steps_between_cache_compressions 128 \
            --p_attn ${p_attn} 
    done
  done
done