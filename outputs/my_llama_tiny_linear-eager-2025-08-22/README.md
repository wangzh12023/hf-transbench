---
library_name: transformers
base_model: configs/my_llama_tiny_sigmoid.json
tags:
- generated_from_trainer
datasets:
- wikitext
metrics:
- accuracy
model-index:
- name: my_llama_tiny_linear-eager-2025-08-22
  results:
  - task:
      name: Causal Language Modeling
      type: text-generation
    dataset:
      name: wikitext wikitext-103-raw-v1
      type: wikitext
      args: wikitext-103-raw-v1
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.0
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# my_llama_tiny_linear-eager-2025-08-22

This model is a fine-tuned version of [configs/my_llama_tiny_sigmoid.json](https://huggingface.co/configs/my_llama_tiny_sigmoid.json) on the wikitext wikitext-103-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: nan
- Accuracy: 0.0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0003
- train_batch_size: 32
- eval_batch_size: 16
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.015
- num_epochs: 1.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Accuracy |
|:-------------:|:------:|:----:|:---------------:|:--------:|
| 0.0           | 0.0647 | 200  | nan             | 0.0      |
| 0.0           | 0.1294 | 400  | nan             | 0.0      |
| 0.0           | 0.1940 | 600  | nan             | 0.0      |
| 0.0           | 0.2587 | 800  | nan             | 0.0      |
| 0.0           | 0.3234 | 1000 | nan             | 0.0      |
| 0.0           | 0.3881 | 1200 | nan             | 0.0      |
| 0.0           | 0.4528 | 1400 | nan             | 0.0      |
| 0.0           | 0.5175 | 1600 | nan             | 0.0      |
| 0.0           | 0.5821 | 1800 | nan             | 0.0      |
| 0.0           | 0.6468 | 2000 | nan             | 0.0      |
| 0.0           | 0.7115 | 2200 | nan             | 0.0      |
| 0.0           | 0.7762 | 2400 | nan             | 0.0      |
| 0.0           | 0.8409 | 2600 | nan             | 0.0      |
| 0.0           | 0.9056 | 2800 | nan             | 0.0      |
| 0.0           | 0.9702 | 3000 | nan             | 0.0      |


### Framework versions

- Transformers 4.47.1
- Pytorch 2.7.1+cu126
- Datasets 3.6.0
- Tokenizers 0.21.2
