---
library_name: transformers
base_model: configs/my_llama_tiny_linear.json
tags:
- generated_from_trainer
datasets:
- wikitext
metrics:
- accuracy
model-index:
- name: my-llama-tiny_linear-2025-07-09
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
      value: 0.053610776131538224
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# my-llama-tiny_linear-2025-07-09

This model is a fine-tuned version of [configs/my_llama_tiny_linear.json](https://huggingface.co/configs/my_llama_tiny_linear.json) on the wikitext wikitext-103-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: 6.6097
- Accuracy: 0.0536

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
| 7.0868        | 0.0235 | 200  | 7.0553          | 0.0475   |
| 7.0107        | 0.0470 | 400  | 6.9734          | 0.0453   |
| 6.9751        | 0.0706 | 600  | 6.9392          | 0.0479   |
| 6.9642        | 0.0941 | 800  | 6.9291          | 0.0475   |
| 6.9519        | 0.1176 | 1000 | 6.9176          | 0.0471   |
| 6.9418        | 0.1411 | 1200 | 6.9029          | 0.0481   |
| 6.932         | 0.1647 | 1400 | 6.8848          | 0.0488   |
| 6.9161        | 0.1882 | 1600 | 6.8667          | 0.0494   |
| 6.92          | 0.2117 | 1800 | 6.8560          | 0.0489   |
| 6.8819        | 0.2352 | 2000 | 6.8368          | 0.0498   |
| 6.8685        | 0.2588 | 2200 | 6.8301          | 0.0492   |
| 6.8672        | 0.2823 | 2400 | 6.8124          | 0.0494   |
| 6.8359        | 0.3058 | 2600 | 6.8060          | 0.0487   |
| 6.8351        | 0.3293 | 2800 | 6.7854          | 0.0489   |
| 6.7913        | 0.3529 | 3000 | 6.7701          | 0.0497   |
| 6.7794        | 0.3764 | 3200 | 6.7575          | 0.0497   |
| 6.7988        | 0.3999 | 3400 | 6.7464          | 0.0505   |
| 6.767         | 0.4234 | 3600 | 6.7323          | 0.0505   |
| 6.7554        | 0.4470 | 3800 | 6.7249          | 0.0507   |
| 6.7676        | 0.4705 | 4000 | 6.7108          | 0.0515   |
| 6.7518        | 0.4940 | 4200 | 6.7009          | 0.0513   |
| 6.7173        | 0.5175 | 4400 | 6.6939          | 0.0515   |
| 6.7093        | 0.5410 | 4600 | 6.6830          | 0.0513   |
| 6.7267        | 0.5646 | 4800 | 6.6820          | 0.0513   |
| 6.714         | 0.5881 | 5000 | 6.6724          | 0.0518   |
| 6.6993        | 0.6116 | 5200 | 6.6663          | 0.0515   |
| 6.7041        | 0.6351 | 5400 | 6.6570          | 0.0520   |
| 6.6528        | 0.6587 | 5600 | 6.6500          | 0.0521   |
| 6.6904        | 0.6822 | 5800 | 6.6454          | 0.0526   |
| 6.7           | 0.7057 | 6000 | 6.6429          | 0.0526   |
| 6.6842        | 0.7292 | 6200 | 6.6353          | 0.0530   |
| 6.6663        | 0.7528 | 6400 | 6.6283          | 0.0529   |
| 6.6353        | 0.7763 | 6600 | 6.6239          | 0.0531   |
| 6.6534        | 0.7998 | 6800 | 6.6226          | 0.0531   |
| 6.6723        | 0.8233 | 7000 | 6.6179          | 0.0532   |
| 6.6486        | 0.8469 | 7200 | 6.6162          | 0.0531   |
| 6.6633        | 0.8704 | 7400 | 6.6136          | 0.0532   |
| 6.6576        | 0.8939 | 7600 | 6.6121          | 0.0534   |
| 6.6704        | 0.9174 | 7800 | 6.6108          | 0.0535   |
| 6.6465        | 0.9410 | 8000 | 6.6102          | 0.0535   |
| 6.6319        | 0.9645 | 8200 | 6.6097          | 0.0536   |
| 6.6234        | 0.9880 | 8400 | 6.6097          | 0.0536   |


### Framework versions

- Transformers 4.47.1
- Pytorch 2.5.1
- Datasets 3.5.0
- Tokenizers 0.21.1
