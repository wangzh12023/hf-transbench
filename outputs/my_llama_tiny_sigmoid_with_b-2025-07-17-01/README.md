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
- name: my_llama_tiny_sigmoid_with_b-2025-07-17-01
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
      value: 0.9997847779967955
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# my_llama_tiny_sigmoid_with_b-2025-07-17-01

This model is a fine-tuned version of [configs/my_llama_tiny_sigmoid_with_b.json](https://huggingface.co/configs/my_llama_tiny_sigmoid_with_b.json) on the wikitext wikitext-103-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0024
- Accuracy: 0.9998

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
| 5.6785        | 0.0235 | 200  | 5.5240          | 0.2029   |
| 5.1636        | 0.0470 | 400  | 5.0980          | 0.2196   |
| 4.9815        | 0.0706 | 600  | 4.9163          | 0.2249   |
| 4.9044        | 0.0941 | 800  | 4.8273          | 0.2260   |
| 4.8084        | 0.1176 | 1000 | 4.7418          | 0.2309   |
| 4.7472        | 0.1411 | 1200 | 4.6755          | 0.2332   |
| 4.6776        | 0.1647 | 1400 | 4.5773          | 0.2405   |
| 4.5732        | 0.1882 | 1600 | 4.4842          | 0.2479   |
| 4.4032        | 0.2117 | 1800 | 4.2466          | 0.2781   |
| 2.7961        | 0.2352 | 2000 | 2.3373          | 0.6090   |
| 0.9324        | 0.2588 | 2200 | 0.7345          | 0.9025   |
| 0.2079        | 0.2823 | 2400 | 0.1507          | 0.9864   |
| 0.0563        | 0.3058 | 2600 | 0.0485          | 0.9964   |
| 0.0387        | 0.3293 | 2800 | 0.0309          | 0.9979   |
| 0.029         | 0.3529 | 3000 | 0.0229          | 0.9984   |
| 0.0164        | 0.3764 | 3200 | 0.0143          | 0.9990   |
| 0.0099        | 0.3999 | 3400 | 0.0104          | 0.9993   |
| 0.0081        | 0.4234 | 3600 | 0.0086          | 0.9994   |
| 0.0081        | 0.4470 | 3800 | 0.0084          | 0.9994   |
| 0.0104        | 0.4705 | 4000 | 0.0087          | 0.9994   |
| 0.0102        | 0.4940 | 4200 | 0.0127          | 0.9990   |
| 0.0082        | 0.5175 | 4400 | 0.0066          | 0.9995   |
| 0.0044        | 0.5410 | 4600 | 0.0049          | 0.9996   |
| 0.0043        | 0.5646 | 4800 | 0.0046          | 0.9996   |
| 0.0038        | 0.5881 | 5000 | 0.0043          | 0.9996   |
| 0.0062        | 0.6116 | 5200 | 0.0046          | 0.9996   |
| 0.0027        | 0.6351 | 5400 | 0.0038          | 0.9996   |
| 0.0027        | 0.6587 | 5600 | 0.0036          | 0.9997   |
| 0.003         | 0.6822 | 5800 | 0.0034          | 0.9997   |
| 0.0029        | 0.7057 | 6000 | 0.0031          | 0.9997   |
| 0.0023        | 0.7292 | 6200 | 0.0031          | 0.9997   |
| 0.0026        | 0.7528 | 6400 | 0.0031          | 0.9998   |
| 0.0023        | 0.7763 | 6600 | 0.0029          | 0.9998   |
| 0.0023        | 0.7998 | 6800 | 0.0027          | 0.9998   |
| 0.0019        | 0.8233 | 7000 | 0.0026          | 0.9998   |
| 0.0019        | 0.8469 | 7200 | 0.0026          | 0.9998   |
| 0.0018        | 0.8704 | 7400 | 0.0025          | 0.9998   |
| 0.0021        | 0.8939 | 7600 | 0.0025          | 0.9998   |
| 0.0022        | 0.9174 | 7800 | 0.0024          | 0.9998   |
| 0.0023        | 0.9410 | 8000 | 0.0024          | 0.9998   |
| 0.0017        | 0.9645 | 8200 | 0.0024          | 0.9998   |
| 0.0016        | 0.9880 | 8400 | 0.0024          | 0.9998   |


### Framework versions

- Transformers 4.47.1
- Pytorch 2.5.1
- Datasets 3.5.0
- Tokenizers 0.21.1
