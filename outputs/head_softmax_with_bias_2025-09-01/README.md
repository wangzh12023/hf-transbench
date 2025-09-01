---
library_name: transformers
base_model: configs/head_softmax.json
tags:
- generated_from_trainer
datasets:
- wikitext
metrics:
- accuracy
model-index:
- name: head_softmax_with_bias_2025-09-01
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
      value: 0.06308737671707872
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# head_softmax_with_bias_2025-09-01

This model is a fine-tuned version of [configs/head_softmax.json](https://huggingface.co/configs/head_softmax.json) on the wikitext wikitext-103-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: 7.0758
- Accuracy: 0.0631

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
| 7.1088        | 0.0441 | 200  | 7.0970          | 0.0636   |
| 7.0807        | 0.0882 | 400  | 7.0796          | 0.0629   |
| 7.1002        | 0.1323 | 600  | 7.0758          | 0.0631   |
| 7.0846        | 0.1764 | 800  | 7.0804          | 0.0630   |
| 7.0892        | 0.2206 | 1000 | 7.0852          | 0.0630   |
| 7.1001        | 0.2647 | 1200 | 7.0922          | 0.0631   |
| 7.1053        | 0.3088 | 1400 | 7.0985          | 0.0630   |
| 7.089         | 0.3529 | 1600 | 7.1057          | 0.0630   |
| 7.1139        | 0.3970 | 1800 | 7.1116          | 0.0630   |
| 7.107         | 0.4411 | 2000 | 7.1156          | 0.0631   |
| 7.1134        | 0.4852 | 2200 | 7.1201          | 0.0631   |
| 7.135         | 0.5293 | 2400 | 7.1234          | 0.0631   |
| 7.1198        | 0.5734 | 2600 | 7.1261          | 0.0630   |
| 7.1139        | 0.6176 | 2800 | 7.1275          | 0.0631   |
| 7.1463        | 0.6617 | 3000 | 7.1289          | 0.0630   |
| 7.1386        | 0.7058 | 3200 | 7.1297          | 0.0630   |
| 7.1212        | 0.7499 | 3400 | 7.1293          | 0.0629   |
| 7.1517        | 0.7940 | 3600 | 7.1294          | 0.0630   |
| 7.1411        | 0.8381 | 3800 | 7.1294          | 0.0630   |
| 7.124         | 0.8822 | 4000 | 7.1293          | 0.0629   |
| 7.1291        | 0.9263 | 4200 | 7.1291          | 0.0629   |
| 7.1386        | 0.9704 | 4400 | 7.1292          | 0.0630   |


### Framework versions

- Transformers 4.47.1
- Pytorch 2.7.1+cu126
- Datasets 3.6.0
- Tokenizers 0.21.2
