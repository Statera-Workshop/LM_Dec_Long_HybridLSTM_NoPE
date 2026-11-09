# LM_Dec_Long_HybridLSTM_NoPE

HybridLSTM_NoPE - Long family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked (5 seed-repeat runs, see results)**
- Failure class: Note: learns more slowly than the GRU version (short runs 1/3 and 1/2), long runs 2/2 exact at 8192

## Provenance

- Source repository: (new) hybrid: LSTM layer + x-transformers decoder
- Upstream / license: lucidrains/x-transformers (MIT), pip 2.31.7 + torch.nn

## Differences from the source ([Unified] protocol)

- ablation of hyb_gru_nope: LSTM instead of GRU in front of the NoPE decoder

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 21.26 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.738 -> 0.001 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.420 / test 1.407 bits per byte
- LM-02 enwik8 (1B tokens): valid 1.332 / test 1.324 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 378 MB, inference 118 MB
- Peak GPU memory at length 8192 (batch 1): train 1367 MB, inference 370 MB
- LM-05 recall `lm05v3`: 4 pairs @ 512: 1.00 (keys removed 0.21); 4 pairs @ 2048: 1.00 (keys removed 0.17); 16 pairs @ 512: 0.06 (keys removed 0.07); 16 pairs @ 2048: 0.06 (keys removed 0.06)
- LM-05 recall `lm05v3long`: 4 pairs @ 512: 1.00 (keys removed 0.19); 4 pairs @ 2048: 1.00 (keys removed 0.15); 4 pairs @ 8192: 1.00 (keys removed 0.13); 16 pairs @ 512: 1.00 (keys removed 0.02); 16 pairs @ 2048: 1.00 (keys removed 0.02); 16 pairs @ 8192: 1.00 (keys removed 0.02)
- LM-05 recall `lm05v3s2`: 4 pairs @ 512: 1.00 (keys removed 0.26); 4 pairs @ 2048: 1.00 (keys removed 0.18); 16 pairs @ 512: 0.07 (keys removed 0.06); 16 pairs @ 2048: 0.07 (keys removed 0.07)
- LM-05 recall `lm05v3s3`: 4 pairs @ 512: 0.99 (keys removed 0.17); 4 pairs @ 2048: 1.00 (keys removed 0.16); 16 pairs @ 512: 1.00 (keys removed 0.05); 16 pairs @ 2048: 1.00 (keys removed 0.04)
- LM-05 recall `lm05v3longs2`: 4 pairs @ 512: 1.00 (keys removed 0.12); 4 pairs @ 2048: 1.00 (keys removed 0.13); 4 pairs @ 8192: 1.00 (keys removed 0.11); 16 pairs @ 512: 1.00 (keys removed 0.02); 16 pairs @ 2048: 1.00 (keys removed 0.02); 16 pairs @ 8192: 1.00 (keys removed 0.01)
- LM-06 memory update `lm06`: @512: latest 0.72, unchanged 0.06, stale 0.00; @2048: latest 0.71, unchanged 0.04, stale 0.01
- LM-06 memory update `lm06long`: @512: latest 1.00, unchanged 1.00, stale 0.00; @2048: latest 1.00, unchanged 1.00, stale 0.00; @8192: latest 1.00, unchanged 1.00, stale 0.00
- LM-06 memory update `lm06s2`: @512: latest 0.99, unchanged 1.00, stale 0.01; @2048: latest 1.00, unchanged 0.99, stale 0.00
- LM-06 memory update `lm06longs2`: @512: latest 1.00, unchanged 1.00, stale 0.00; @2048: latest 1.00, unchanged 1.00, stale 0.00; @8192: latest 1.00, unchanged 1.00, stale 0.00

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check hyb_lstm_nope
python run.py lm02 hyb_lstm_nope      # enwik8 is downloaded to data/ on first use
```
