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
- name: my_llama_tiny_sigmoid-eager-2025-08-21
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
      value: 0.05942860266260364
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# my_llama_tiny_sigmoid-eager-2025-08-21

This model is a fine-tuned version of [configs/my_llama_tiny_sigmoid.json](https://huggingface.co/configs/my_llama_tiny_sigmoid.json) on the wikitext wikitext-103-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: 6.4686
- Accuracy: 0.0594

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
| 7.0733        | 0.0441 | 200  | 7.0222          | 0.0462   |
| 7.009         | 0.0882 | 400  | 6.9641          | 0.0485   |
| 6.9539        | 0.1323 | 600  | 6.9136          | 0.0497   |
| 6.9297        | 0.1764 | 800  | 6.8685          | 0.0504   |
| 6.8734        | 0.2206 | 1000 | 6.8497          | 0.0508   |
| 6.8461        | 0.2647 | 1200 | 6.8062          | 0.0506   |
| 6.7896        | 0.3088 | 1400 | 6.7665          | 0.0516   |
| 6.7535        | 0.3529 | 1600 | 6.7305          | 0.0523   |
| 6.7449        | 0.3970 | 1800 | 6.7035          | 0.0521   |
| 6.7104        | 0.4411 | 2000 | 6.6681          | 0.0534   |
| 6.6798        | 0.4852 | 2200 | 6.6374          | 0.0545   |
| 6.6436        | 0.5293 | 2400 | 6.5980          | 0.0556   |
| 6.5996        | 0.5734 | 2600 | 6.5756          | 0.0564   |
| 6.5893        | 0.6176 | 2800 | 6.5488          | 0.0571   |
| 6.5466        | 0.6617 | 3000 | 6.5269          | 0.0578   |
| 6.5569        | 0.7058 | 3200 | 6.5115          | 0.0584   |
| 6.529         | 0.7499 | 3400 | 6.4969          | 0.0588   |
| 6.5428        | 0.7940 | 3600 | 6.4863          | 0.0589   |
| 6.5166        | 0.8381 | 3800 | 6.4785          | 0.0594   |
| 6.5266        | 0.8822 | 4000 | 6.4725          | 0.0593   |
| 6.5253        | 0.9263 | 4200 | 6.4702          | 0.0594   |
| 6.4967        | 0.9704 | 4400 | 6.4686          | 0.0594   |


### Framework versions

- Transformers 4.47.1
- Pytorch 2.7.1+cu126
- Datasets 3.6.0
- Tokenizers 0.21.2
