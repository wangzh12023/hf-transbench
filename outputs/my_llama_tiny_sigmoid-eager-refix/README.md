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
- name: my_llama_tiny_sigmoid-eager-refix
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
      value: 0.9638187899057464
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# my_llama_tiny_sigmoid-eager-refix

This model is a fine-tuned version of [configs/my_llama_tiny_sigmoid.json](https://huggingface.co/configs/my_llama_tiny_sigmoid.json) on the wikitext wikitext-103-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2021
- Accuracy: 0.9638

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
| 5.529         | 0.0441 | 200  | 5.3601          | 0.2086   |
| 5.0685        | 0.0882 | 400  | 4.9930          | 0.2208   |
| 4.9084        | 0.1323 | 600  | 4.8413          | 0.2242   |
| 4.8361        | 0.1764 | 800  | 4.7402          | 0.2281   |
| 4.7194        | 0.2206 | 1000 | 4.6461          | 0.2345   |
| 4.5794        | 0.2647 | 1200 | 4.4427          | 0.2652   |
| 3.1684        | 0.3088 | 1400 | 2.9534          | 0.5213   |
| 2.3577        | 0.3529 | 1600 | 2.2753          | 0.6203   |
| 1.7635        | 0.3970 | 1800 | 1.6573          | 0.7060   |
| 1.3976        | 0.4411 | 2000 | 1.3351          | 0.7568   |
| 1.1627        | 0.4852 | 2200 | 1.0828          | 0.8012   |
| 0.8214        | 0.5293 | 2400 | 0.7641          | 0.8624   |
| 0.5984        | 0.5734 | 2600 | 0.5673          | 0.9019   |
| 0.5064        | 0.6176 | 2800 | 0.4798          | 0.9172   |
| 0.4271        | 0.6617 | 3000 | 0.4109          | 0.9291   |
| 0.3729        | 0.7058 | 3200 | 0.3568          | 0.9381   |
| 0.321         | 0.7499 | 3400 | 0.3073          | 0.9461   |
| 0.2785        | 0.7940 | 3600 | 0.2660          | 0.9528   |
| 0.2441        | 0.8381 | 3800 | 0.2325          | 0.9584   |
| 0.225         | 0.8822 | 4000 | 0.2143          | 0.9618   |
| 0.2136        | 0.9263 | 4200 | 0.2050          | 0.9633   |
| 0.2089        | 0.9704 | 4400 | 0.2021          | 0.9638   |


### Framework versions

- Transformers 4.47.1
- Pytorch 2.7.1+cu126
- Datasets 3.6.0
- Tokenizers 0.21.2
