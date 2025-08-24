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
- name: head_softmax
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
      value: 0.05864287153979387
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# head_softmax

This model is a fine-tuned version of [configs/head_softmax.json](https://huggingface.co/configs/head_softmax.json) on the wikitext wikitext-103-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: 7.0060
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
| 7.0592        | 0.0500 | 200  | 7.0248          | 0.0587   |
| 7.035         | 0.1000 | 400  | 7.0225          | 0.0589   |
| 7.0514        | 0.1500 | 600  | 7.0107          | 0.0588   |
| 7.0395        | 0.2000 | 800  | 7.0086          | 0.0588   |
| 7.0408        | 0.2499 | 1000 | 7.0060          | 0.0586   |
| 7.0559        | 0.2999 | 1200 | 7.0106          | 0.0585   |
| 7.0433        | 0.3499 | 1400 | 7.0131          | 0.0583   |
| 7.0556        | 0.3999 | 1600 | 7.0164          | 0.0582   |
| 7.041         | 0.4499 | 1800 | 7.0180          | 0.0583   |
| 7.042         | 0.4999 | 2000 | 7.0173          | 0.0584   |
| 7.0546        | 0.5499 | 2200 | 7.0154          | 0.0586   |
| 7.0584        | 0.5999 | 2400 | 7.0138          | 0.0586   |
| 7.043         | 0.6498 | 2600 | 7.0138          | 0.0585   |
| 7.0537        | 0.6998 | 2800 | 7.0139          | 0.0585   |
| 7.0469        | 0.7498 | 3000 | 7.0138          | 0.0586   |
| 7.0675        | 0.7998 | 3200 | 7.0142          | 0.0585   |
| 7.0615        | 0.8498 | 3400 | 7.0140          | 0.0585   |
| 7.0528        | 0.8998 | 3600 | 7.0141          | 0.0585   |
| 7.0574        | 0.9498 | 3800 | 7.0141          | 0.0585   |
| 7.0456        | 0.9998 | 4000 | 7.0141          | 0.0586   |


### Framework versions

- Transformers 4.47.1
- Pytorch 2.7.1+cu126
- Datasets 3.6.0
- Tokenizers 0.21.2
