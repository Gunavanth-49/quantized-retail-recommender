# Quantized Retail Recommender

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/w2pbxh5p6a/badge.svg)](https://ask.thili.ai/b/w2pbxh5p6a)

A Streamlit app that recommends the next item in a shopping basket and measures how much INT8 quantization shrinks a small language model. The experiment runs locally using synthetic data.

## Run with Python 3.12

From this directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

Open the local URL printed by Streamlit. Select basket items and a model in **Basket recommendations**. View the benchmark and download an NPZ checkpoint in **Model comparison**.

The app runs without an API key, Ollama, or a model download. The starter's optional chat integration still reads its settings from `.env`. Keep that file out of Git. The original GitHub Actions workflow installs the dependencies and runs pytest on pushes and pull requests.

## Milestones

1. Generate 800 source, 1,000 target training, 200 calibration, and 300 test baskets with seeded random streams. The target store favors popular items in each shopping theme.
2. Pretrain a two-layer causal next-token model on source basket prefixes, then continue training from that checkpoint on the target training baskets. Each product is a token. The model pools the observed items so recommendations are invariant to input order.
3. Rank unseen products with the model. Itemset support is the fraction of target training baskets containing every selected item. Co-purchase confidence is the fraction of those matching baskets that also contain the suggested product.
4. Evaluate FP32 top-1 and top-5 accuracy by predicting each next item from its preceding prefix. Previously seen items are masked. Count the actual bytes in the downloadable, uncompressed NPZ checkpoint.
5. Apply post-training INT8 quantization to both dense layers. Dynamic quantization computes a fresh activation scale for each input row. Static quantization freezes activation scales estimated on the calibration split. Both use per-output-channel INT8 weights and INT32 accumulation. Biases and the tanh activation use floating point.
6. Evaluate both quantized variants on the same held-out targets and report checkpoint size reduction and accuracy changes in percentage points.

## Reference results

Default seed 42 produces 952 held-out next-item predictions. The source checkpoint scores 12.71% top-1 before continued pretraining.

| Model | Checkpoint bytes | Size reduction | Top-1 accuracy | Top-5 accuracy |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 34,374 | 0% | 33.09% | 94.33% |
| Dynamic INT8 | 10,934 | 68.19% | 33.40% | 94.33% |
| Static INT8 | 11,452 | 66.68% | 33.30% | 94.43% |

These results measure synthetic-data accuracy and serialized checkpoint size. The NumPy reference implementation provides integer accumulation. Inference latency and real retail performance are unmeasured.
