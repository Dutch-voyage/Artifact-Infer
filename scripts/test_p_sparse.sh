for p_attn in 0.80 0.90; do
  for layer_budget in 1024 2048 4096; do
    for window_size in 32; do
      for steps_between_cache_compressions in 32; do

        python -m eval.test_aime_evict \
            --compress_method rkv \
            --if_fake_compress True \
            --layer_budget ${layer_budget} \
            --window_size ${window_size} \
            --steps_between_cache_compressions ${steps_between_cache_compressions} \
            --p_attn ${p_attn} 
      done
    done
  done
done