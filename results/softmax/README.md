---
library_name: transformers
base_model: configs/my_llama_tiny.json
tags:
- generated_from_trainer
datasets:
- wikitext
metrics:
- accuracy
model-index:
- name: softmax
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
      value: 0.42977101062103507
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# softmax

This model is a fine-tuned version of [configs/my_llama_tiny.json](https://huggingface.co/configs/my_llama_tiny.json) on the wikitext wikitext-103-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: 3.1635
- Accuracy: 0.4298

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
| 5.3666        | 0.0500 | 200  | 5.1748          | 0.2494   |
| 4.6675        | 0.1000 | 400  | 4.5418          | 0.2903   |
| 4.3438        | 0.1500 | 600  | 4.2316          | 0.3128   |
| 4.125         | 0.2000 | 800  | 4.0257          | 0.3299   |
| 3.9706        | 0.2499 | 1000 | 3.8638          | 0.3458   |
| 3.8622        | 0.2999 | 1200 | 3.7511          | 0.3565   |
| 3.737         | 0.3499 | 1400 | 3.6547          | 0.3673   |
| 3.6631        | 0.3999 | 1600 | 3.5626          | 0.3789   |
| 3.5527        | 0.4499 | 1800 | 3.4728          | 0.3921   |
| 3.4868        | 0.4999 | 2000 | 3.3971          | 0.4013   |
| 3.4251        | 0.5499 | 2200 | 3.3391          | 0.4071   |
| 3.3924        | 0.5999 | 2400 | 3.2943          | 0.4132   |
| 3.3429        | 0.6498 | 2600 | 3.2568          | 0.4169   |
| 3.3224        | 0.6998 | 2800 | 3.2272          | 0.4216   |
| 3.2858        | 0.7498 | 3000 | 3.2040          | 0.4244   |
| 3.2765        | 0.7998 | 3200 | 3.1867          | 0.4264   |
| 3.2541        | 0.8498 | 3400 | 3.1749          | 0.4283   |
| 3.2384        | 0.8998 | 3600 | 3.1674          | 0.4291   |
| 3.2455        | 0.9498 | 3800 | 3.1640          | 0.4297   |
| 3.2393        | 0.9998 | 4000 | 3.1635          | 0.4298   |


### Framework versions

- Transformers 4.47.1
- Pytorch 2.7.1+cu126
- Datasets 3.6.0
- Tokenizers 0.21.2
