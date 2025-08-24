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
- name: sigmoid
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
      value: 0.349028597196648
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sigmoid

This model is a fine-tuned version of [configs/my_llama_tiny_sigmoid.json](https://huggingface.co/configs/my_llama_tiny_sigmoid.json) on the wikitext wikitext-103-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: 3.7446
- Accuracy: 0.3490

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
| 6.1888        | 0.0500 | 200  | 5.9668          | 0.1585   |
| 5.5124        | 0.1000 | 400  | 5.4063          | 0.2004   |
| 5.1397        | 0.1500 | 600  | 5.0242          | 0.2378   |
| 4.8352        | 0.2000 | 800  | 4.7386          | 0.2613   |
| 4.624         | 0.2499 | 1000 | 4.5214          | 0.2793   |
| 4.4785        | 0.2999 | 1200 | 4.3664          | 0.2900   |
| 4.3355        | 0.3499 | 1400 | 4.2498          | 0.2999   |
| 4.2551        | 0.3999 | 1600 | 4.1544          | 0.3080   |
| 4.1482        | 0.4499 | 1800 | 4.0689          | 0.3152   |
| 4.0875        | 0.4999 | 2000 | 3.9963          | 0.3230   |
| 4.0282        | 0.5499 | 2200 | 3.9394          | 0.3278   |
| 3.9836        | 0.5999 | 2400 | 3.8907          | 0.3328   |
| 3.9271        | 0.6498 | 2600 | 3.8471          | 0.3376   |
| 3.9055        | 0.6998 | 2800 | 3.8171          | 0.3406   |
| 3.8664        | 0.7498 | 3000 | 3.7895          | 0.3443   |
| 3.854         | 0.7998 | 3200 | 3.7703          | 0.3457   |
| 3.8367        | 0.8498 | 3400 | 3.7571          | 0.3476   |
| 3.8139        | 0.8998 | 3600 | 3.7490          | 0.3486   |
| 3.8291        | 0.9498 | 3800 | 3.7453          | 0.3492   |
| 3.8191        | 0.9998 | 4000 | 3.7446          | 0.3490   |


### Framework versions

- Transformers 4.47.1
- Pytorch 2.7.1+cu126
- Datasets 3.6.0
- Tokenizers 0.21.2
