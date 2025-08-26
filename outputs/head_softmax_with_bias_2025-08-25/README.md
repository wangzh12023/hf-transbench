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
- name: head_softmax_with_bias_2025-08-25
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
      value: 0.058632622872974605
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# head_softmax_with_bias_2025-08-25

This model is a fine-tuned version of [configs/head_softmax.json](https://huggingface.co/configs/head_softmax.json) on the wikitext wikitext-103-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: 7.0218
- Accuracy: 0.0586

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
| 7.0802        | 0.0441 | 200  | 7.0397          | 0.0595   |
| 7.0601        | 0.0882 | 400  | 7.0299          | 0.0599   |
| 7.0697        | 0.1323 | 600  | 7.0256          | 0.0594   |
| 7.0632        | 0.1764 | 800  | 7.0221          | 0.0591   |
| 7.042         | 0.2206 | 1000 | 7.0218          | 0.0586   |
| 7.0514        | 0.2647 | 1200 | 7.0264          | 0.0586   |
| 7.0626        | 0.3088 | 1400 | 7.0320          | 0.0584   |
| 7.0414        | 0.3529 | 1600 | 7.0308          | 0.0581   |
| 7.0726        | 0.3970 | 1800 | 7.0356          | 0.0583   |
| 7.0599        | 0.4411 | 2000 | 7.0391          | 0.0583   |
| 7.0576        | 0.4852 | 2200 | 7.0416          | 0.0581   |
| 7.0676        | 0.5293 | 2400 | 7.0413          | 0.0580   |
| 7.0618        | 0.5734 | 2600 | 7.0416          | 0.0580   |
| 7.0632        | 0.6176 | 2800 | 7.0425          | 0.0580   |
| 7.0765        | 0.6617 | 3000 | 7.0424          | 0.0578   |
| 7.0926        | 0.7058 | 3200 | 7.0424          | 0.0578   |
| 7.0581        | 0.7499 | 3400 | 7.0423          | 0.0579   |
| 7.0983        | 0.7940 | 3600 | 7.0418          | 0.0579   |
| 7.078         | 0.8381 | 3800 | 7.0419          | 0.0579   |
| 7.0712        | 0.8822 | 4000 | 7.0414          | 0.0579   |
| 7.0746        | 0.9263 | 4200 | 7.0416          | 0.0579   |
| 7.0762        | 0.9704 | 4400 | 7.0415          | 0.0579   |


### Framework versions

- Transformers 4.47.1
- Pytorch 2.7.1+cu126
- Datasets 3.6.0
- Tokenizers 0.21.2
