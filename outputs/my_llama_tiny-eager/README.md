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
- name: my_llama_tiny-eager
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
      value: 0.44726548488150836
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# my_llama_tiny-eager

This model is a fine-tuned version of [configs/my_llama_tiny.json](https://huggingface.co/configs/my_llama_tiny.json) on the wikitext wikitext-103-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: 2.9938
- Accuracy: 0.4473

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
| 5.6989        | 0.0235 | 200  | 5.5289          | 0.2207   |
| 4.9574        | 0.0470 | 400  | 4.8622          | 0.2669   |
| 4.6057        | 0.0706 | 600  | 4.5174          | 0.2918   |
| 4.4086        | 0.0941 | 800  | 4.3171          | 0.3044   |
| 4.2391        | 0.1176 | 1000 | 4.1561          | 0.3169   |
| 4.1343        | 0.1411 | 1200 | 4.0451          | 0.3276   |
| 4.0564        | 0.1647 | 1400 | 3.9387          | 0.3375   |
| 3.9485        | 0.1882 | 1600 | 3.8624          | 0.3435   |
| 3.9073        | 0.2117 | 1800 | 3.7808          | 0.3524   |
| 3.8174        | 0.2352 | 2000 | 3.7023          | 0.3612   |
| 3.7391        | 0.2588 | 2200 | 3.6474          | 0.3659   |
| 3.6724        | 0.2823 | 2400 | 3.5750          | 0.3772   |
| 3.5882        | 0.3058 | 2600 | 3.5099          | 0.3856   |
| 3.539         | 0.3293 | 2800 | 3.4536          | 0.3928   |
| 3.4676        | 0.3529 | 3000 | 3.4073          | 0.3988   |
| 3.4309        | 0.3764 | 3200 | 3.3627          | 0.4040   |
| 3.4178        | 0.3999 | 3400 | 3.3252          | 0.4081   |
| 3.3553        | 0.4234 | 3600 | 3.2892          | 0.4118   |
| 3.3289        | 0.4470 | 3800 | 3.2629          | 0.4146   |
| 3.3316        | 0.4705 | 4000 | 3.2322          | 0.4192   |
| 3.3109        | 0.4940 | 4200 | 3.2087          | 0.4212   |
| 3.2599        | 0.5175 | 4400 | 3.1820          | 0.4238   |
| 3.2204        | 0.5410 | 4600 | 3.1633          | 0.4269   |
| 3.2266        | 0.5646 | 4800 | 3.1446          | 0.4289   |
| 3.2302        | 0.5881 | 5000 | 3.1270          | 0.4303   |
| 3.1676        | 0.6116 | 5200 | 3.1101          | 0.4326   |
| 3.1922        | 0.6351 | 5400 | 3.0934          | 0.4349   |
| 3.1185        | 0.6587 | 5600 | 3.0778          | 0.4364   |
| 3.1593        | 0.6822 | 5800 | 3.0655          | 0.4380   |
| 3.1469        | 0.7057 | 6000 | 3.0538          | 0.4397   |
| 3.1411        | 0.7292 | 6200 | 3.0442          | 0.4409   |
| 3.1042        | 0.7528 | 6400 | 3.0325          | 0.4427   |
| 3.0759        | 0.7763 | 6600 | 3.0240          | 0.4427   |
| 3.0918        | 0.7998 | 6800 | 3.0174          | 0.4444   |
| 3.1194        | 0.8233 | 7000 | 3.0109          | 0.4451   |
| 3.0928        | 0.8469 | 7200 | 3.0062          | 0.4458   |
| 3.0928        | 0.8704 | 7400 | 3.0013          | 0.4463   |
| 3.0704        | 0.8939 | 7600 | 2.9985          | 0.4466   |
| 3.0994        | 0.9174 | 7800 | 2.9962          | 0.4470   |
| 3.0653        | 0.9410 | 8000 | 2.9948          | 0.4471   |
| 3.0504        | 0.9645 | 8200 | 2.9941          | 0.4473   |
| 3.0487        | 0.9880 | 8400 | 2.9938          | 0.4473   |


### Framework versions

- Transformers 4.47.1
- Pytorch 2.5.1
- Datasets 3.5.0
- Tokenizers 0.21.1
