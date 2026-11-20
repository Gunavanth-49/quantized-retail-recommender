# Quantized Retail Recommender

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/w2pbxh5p6a/badge.svg)](https://ask.thili.ai/b/w2pbxh5p6a)

This project develops a retail recommender system capable of identifying frequently bought items from shopping basket data. It then applies post-training quantization techniques to significantly reduce the model's size, making it suitable for deployment on resource-constrained edge devices or for faster loading in applications. The result is an efficient, smart recommender with a minimal footprint.

## What you are building

**Goal.** You will build a smart retail recommender that identifies frequently bought items, then shrink its model using quantization to reduce its footprint for edge device deployment or faster loading.

**Done when.** You will have a functional recommendation model, trained on synthetic shopping basket data, that can predict next items and probe itemsets. This model will then be successfully quantized, demonstrating a measurable reduction in size while maintaining acceptable accuracy.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare synthetic shopping basket data for training**  
      Hint: Use the data preparation methods from `Teach a Language Model What Sells Together`.
- [ ] **2. Continue pretraining a small LM on the basket data**  
      Hint: Apply the continued pretraining techniques from `Teach a Language Model What Sells Together`.
- [ ] **3. Implement next-item prediction and itemset probing**  
      Hint: Implement the prediction and probing methods from `Teach a Language Model What Sells Together`.
- [ ] **4. Measure the baseline size and accuracy of the model**  
      Hint: Evaluate the model's performance and size as shown in `Teach a Language Model What Sells Together`.
- [ ] **5. Apply dynamic and static post-training quantization to the model**  
      Hint: Apply the dynamic and static quantization techniques from `Shrink a Neural Network with Quantisation`.
- [ ] **6. Demonstrate size reduction and evaluate accuracy impact**  
      Hint: Measure the size reduction and accuracy impact as demonstrated in `Shrink a Neural Network with Quantisation`.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Benchmark the inference speed of the original and quantized models to quantify performance gains.
- Deep-dive into the accuracy impact, analyzing if certain types of recommendations (e.g., for rare items) are more affected by quantization.
- Explore different calibration datasets or methods for static quantization, if the drill provides options, to optimize the trade-off between size and accuracy.
