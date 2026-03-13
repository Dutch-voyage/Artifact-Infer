for data_source in umathtop50 aime24 aime25; do
    python -m eval.test_aime_baseline \
        --data_source ${data_source} \
        --if_compress_kvcache False \
        --enforce_eager True 
done