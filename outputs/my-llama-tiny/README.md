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
- name: my-llama-tiny
  results:
  - task:
      name: Causal Language Modeling
      type: text-generation
    dataset:
      name: wikitext wikitext-2-raw-v1
      type: wikitext
      args: wikitext-2-raw-v1
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.15223028071098418
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# my-llama-tiny

This model is a fine-tuned version of [configs/my_llama_tiny.json](https://huggingface.co/configs/my_llama_tiny.json) on the wikitext wikitext-2-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: 6.4263
- Accuracy: 0.1522

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



### Framework versions

- Transformers 4.47.1
- Pytorch 2.5.1
- Datasets 3.5.0
- Tokenizers 0.21.1
