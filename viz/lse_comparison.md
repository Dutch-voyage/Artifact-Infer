# Hyparamerter Explanations
## if_fake_compress = True

When `if_fake_compress=True`, there is no KV Cache compression at all (keeping full KV). 

`lse` means the LogSumExp of existing KV cache, so as long as `if_fake_compress=True`, this term is same, when generation-related hyperparameters are same. 

`lse_topp` means the TopP lse of existing KV cache, this term is always computed in **raw(original)** logits, which is irrelavant to the calibration methods (maxpool/sim, e.t.c) when `if_fake_compress=True`.

`num_topp` means the number of TopP selection tokens, this term is always computed in the **calibrated** logits/attention scores. The difference is in the range of selection. When `if_fake_compress=True`, the selection takes place in full KV.

## if_fake_copmress= False

When `if_fake_compress=False`, the KV cache is compressed under a given budget (fixed budget/ linear budget/integral budget)

### Fixed budget

`lse` means the LogSumExp of existing KV cache, so it is reduced over the compressed KV cache. 

`lse_topp` means the TopP lse of existing KV cache, so it is computed over the compressed KV cache for different compression methods, yet the logits is not calibrated/shifted by different methods. 

`num_topp` means the number of TopP selection tokens, both the selected tokens and the logits/attention scores are manipulated by different compression methods. 
