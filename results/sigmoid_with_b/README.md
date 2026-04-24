---
library_name: transformers
base_model: configs/my_llama_tiny_sigmoid_with_b.json
tags:
- generated_from_trainer
datasets:
- wikitext
metrics:
- accuracy
model-index:
- name: sigmoid_with_b
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
      value: 0.41965899269270057
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sigmoid_with_b

This model is a fine-tuned version of [configs/my_llama_tiny_sigmoid_with_b.json](https://huggingface.co/configs/my_llama_tiny_sigmoid_with_b.json) on the wikitext wikitext-103-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: 3.2254
- Accuracy: 0.4197

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
| 5.2563        | 0.0441 | 200  | 5.0913          | 0.2494   |
| 4.6875        | 0.0882 | 400  | 4.5870          | 0.2786   |
| 4.3965        | 0.1323 | 600  | 4.3104          | 0.2985   |
| 4.2374        | 0.1764 | 800  | 4.1194          | 0.3159   |
| 4.0611        | 0.2206 | 1000 | 3.9800          | 0.3291   |
| 3.9737        | 0.2647 | 1200 | 3.8656          | 0.3403   |
| 3.8399        | 0.3088 | 1400 | 3.7728          | 0.3516   |
| 3.7458        | 0.3529 | 1600 | 3.6864          | 0.3624   |
| 3.7056        | 0.3970 | 1800 | 3.6123          | 0.3714   |
| 3.61          | 0.4411 | 2000 | 3.5373          | 0.3807   |
| 3.5663        | 0.4852 | 2200 | 3.4815          | 0.3881   |
| 3.4981        | 0.5293 | 2400 | 3.4249          | 0.3942   |
| 3.4386        | 0.5734 | 2600 | 3.3861          | 0.3991   |
| 3.4156        | 0.6176 | 2800 | 3.3491          | 0.4033   |
| 3.3753        | 0.6617 | 3000 | 3.3163          | 0.4083   |
| 3.365         | 0.7058 | 3200 | 3.2899          | 0.4111   |
| 3.3381        | 0.7499 | 3400 | 3.2700          | 0.4135   |
| 3.3386        | 0.7940 | 3600 | 3.2529          | 0.4161   |
| 3.313         | 0.8381 | 3800 | 3.2403          | 0.4178   |
| 3.3296        | 0.8822 | 4000 | 3.2322          | 0.4183   |
| 3.3161        | 0.9263 | 4200 | 3.2270          | 0.4191   |
| 3.2863        | 0.9704 | 4400 | 3.2254          | 0.4197   |


### Framework versions

- Transformers 4.47.1
- Pytorch 2.7.1+cu126
- Datasets 3.6.0
- Tokenizers 0.21.2
