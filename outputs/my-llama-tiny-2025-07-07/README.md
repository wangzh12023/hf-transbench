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
- name: my-llama-tiny-2025-07-07
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
      value: 0.43167384642714396
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# my-llama-tiny-2025-07-07

This model is a fine-tuned version of [configs/my_llama_tiny.json](https://huggingface.co/configs/my_llama_tiny.json) on the wikitext wikitext-103-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: 3.1438
- Accuracy: 0.4317

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
| 5.3964        | 0.0470 | 200  | 5.2038          | 0.2465   |
| 4.6861        | 0.0941 | 400  | 4.5656          | 0.2883   |
| 4.3562        | 0.1411 | 600  | 4.2515          | 0.3108   |
| 4.1429        | 0.1882 | 800  | 4.0439          | 0.3278   |
| 3.9893        | 0.2352 | 1000 | 3.8846          | 0.3429   |
| 3.877         | 0.2823 | 1200 | 3.7678          | 0.3550   |
| 3.7585        | 0.3293 | 1400 | 3.6649          | 0.3655   |
| 3.6744        | 0.3764 | 1600 | 3.5824          | 0.3752   |
| 3.5809        | 0.4234 | 1800 | 3.4987          | 0.3862   |
| 3.523         | 0.4705 | 2000 | 3.4212          | 0.3961   |
| 3.4375        | 0.5175 | 2200 | 3.3552          | 0.4053   |
| 3.3991        | 0.5646 | 2400 | 3.3096          | 0.4097   |
| 3.3437        | 0.6116 | 2600 | 3.2673          | 0.4156   |
| 3.2808        | 0.6587 | 2800 | 3.2292          | 0.4207   |
| 3.2896        | 0.7057 | 3000 | 3.2057          | 0.4236   |
| 3.2457        | 0.7528 | 3200 | 3.1828          | 0.4268   |
| 3.2509        | 0.7998 | 3400 | 3.1673          | 0.4285   |
| 3.2364        | 0.8469 | 3600 | 3.1565          | 0.4300   |
| 3.2403        | 0.8939 | 3800 | 3.1481          | 0.4312   |
| 3.208         | 0.9410 | 4000 | 3.1447          | 0.4314   |
| 3.2178        | 0.9880 | 4200 | 3.1438          | 0.4317   |


### Framework versions

- Transformers 4.47.1
- Pytorch 2.5.1
- Datasets 3.5.0
- Tokenizers 0.21.1
