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
- name: my_llama_tiny_sigmoid-eager-2025-08-22
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
      value: 0.3544843041667663
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# my_llama_tiny_sigmoid-eager-2025-08-22

This model is a fine-tuned version of [configs/my_llama_tiny_sigmoid.json](https://huggingface.co/configs/my_llama_tiny_sigmoid.json) on the wikitext wikitext-103-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: 3.6946
- Accuracy: 0.3545

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
| 6.2485        | 0.0441 | 200  | 6.0185          | 0.1570   |
| 5.5486        | 0.0882 | 400  | 5.4515          | 0.1955   |
| 5.1903        | 0.1323 | 600  | 5.0917          | 0.2316   |
| 4.931         | 0.1764 | 800  | 4.8016          | 0.2558   |
| 4.6701        | 0.2206 | 1000 | 4.5847          | 0.2715   |
| 4.5268        | 0.2647 | 1200 | 4.4149          | 0.2849   |
| 4.3629        | 0.3088 | 1400 | 4.2921          | 0.2948   |
| 4.2532        | 0.3529 | 1600 | 4.1943          | 0.3040   |
| 4.2091        | 0.3970 | 1800 | 4.1092          | 0.3120   |
| 4.1151        | 0.4411 | 2000 | 4.0356          | 0.3186   |
| 4.0653        | 0.4852 | 2200 | 3.9695          | 0.3253   |
| 3.9943        | 0.5293 | 2400 | 3.9084          | 0.3311   |
| 3.9296        | 0.5734 | 2600 | 3.8670          | 0.3358   |
| 3.8988        | 0.6176 | 2800 | 3.8274          | 0.3397   |
| 3.8494        | 0.6617 | 3000 | 3.7895          | 0.3437   |
| 3.8417        | 0.7058 | 3200 | 3.7626          | 0.3462   |
| 3.8112        | 0.7499 | 3400 | 3.7387          | 0.3496   |
| 3.8134        | 0.7940 | 3600 | 3.7215          | 0.3513   |
| 3.7837        | 0.8381 | 3800 | 3.7092          | 0.3528   |
| 3.7973        | 0.8822 | 4000 | 3.7010          | 0.3535   |
| 3.7886        | 0.9263 | 4200 | 3.6964          | 0.3539   |
| 3.7619        | 0.9704 | 4400 | 3.6946          | 0.3545   |


### Framework versions

- Transformers 4.47.1
- Pytorch 2.7.1+cu126
- Datasets 3.6.0
- Tokenizers 0.21.2
