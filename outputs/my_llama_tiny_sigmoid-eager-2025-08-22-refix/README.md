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
- name: my_llama_tiny_sigmoid-eager-2025-08-22-refix
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
      value: 0.43338537378596
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# my_llama_tiny_sigmoid-eager-2025-08-22-refix

This model is a fine-tuned version of [configs/my_llama_tiny_sigmoid.json](https://huggingface.co/configs/my_llama_tiny_sigmoid.json) on the wikitext wikitext-103-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: 3.1373
- Accuracy: 0.4334

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
| 5.4424        | 0.0441 | 200  | 5.2469          | 0.2411   |
| 4.7114        | 0.0882 | 400  | 4.5956          | 0.2859   |
| 4.372         | 0.1323 | 600  | 4.2749          | 0.3092   |
| 4.1872        | 0.1764 | 800  | 4.0673          | 0.3267   |
| 4.005         | 0.2206 | 1000 | 3.9256          | 0.3405   |
| 3.9102        | 0.2647 | 1200 | 3.8010          | 0.3521   |
| 3.7639        | 0.3088 | 1400 | 3.6939          | 0.3651   |
| 3.6442        | 0.3529 | 1600 | 3.5877          | 0.3797   |
| 3.5901        | 0.3970 | 1800 | 3.4943          | 0.3910   |
| 3.4955        | 0.4411 | 2000 | 3.4258          | 0.3999   |
| 3.4532        | 0.4852 | 2200 | 3.3659          | 0.4062   |
| 3.392         | 0.5293 | 2400 | 3.3137          | 0.4123   |
| 3.3347        | 0.5734 | 2600 | 3.2767          | 0.4165   |
| 3.3156        | 0.6176 | 2800 | 3.2467          | 0.4203   |
| 3.2759        | 0.6617 | 3000 | 3.2166          | 0.4243   |
| 3.2697        | 0.7058 | 3200 | 3.1948          | 0.4265   |
| 3.2447        | 0.7499 | 3400 | 3.1747          | 0.4291   |
| 3.2488        | 0.7940 | 3600 | 3.1610          | 0.4305   |
| 3.221         | 0.8381 | 3800 | 3.1500          | 0.4317   |
| 3.2428        | 0.8822 | 4000 | 3.1429          | 0.4329   |
| 3.2313        | 0.9263 | 4200 | 3.1389          | 0.4332   |
| 3.1987        | 0.9704 | 4400 | 3.1373          | 0.4334   |


### Framework versions

- Transformers 4.47.1
- Pytorch 2.7.1+cu126
- Datasets 3.6.0
- Tokenizers 0.21.2
