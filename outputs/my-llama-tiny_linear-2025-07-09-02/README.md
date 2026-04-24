---
library_name: transformers
base_model: configs/my_llama_tiny_linear.json
tags:
- generated_from_trainer
datasets:
- wikitext
metrics:
- accuracy
model-index:
- name: my-llama-tiny_linear-2025-07-09-02
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
      value: 0.422696014293474
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# my-llama-tiny_linear-2025-07-09-02

This model is a fine-tuned version of [configs/my_llama_tiny_linear.json](https://huggingface.co/configs/my_llama_tiny_linear.json) on the wikitext wikitext-103-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: 3.3817
- Accuracy: 0.4227

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
| 7.1024        | 0.0235 | 200  | 7.0782          | 0.0424   |
| 7.0887        | 0.0470 | 400  | 7.0582          | 0.0418   |
| 7.0482        | 0.0706 | 600  | 7.0320          | 0.0448   |
| 7.0739        | 0.0941 | 800  | 7.0048          | 0.0448   |
| 7.0342        | 0.1176 | 1000 | 6.9793          | 0.0473   |
| 7.0082        | 0.1411 | 1200 | 6.9713          | 0.0465   |
| 7.0008        | 0.1647 | 1400 | 6.9598          | 0.0466   |
| 6.9897        | 0.1882 | 1600 | 6.9475          | 0.0455   |
| 7.0023        | 0.2117 | 1800 | 6.9398          | 0.0451   |
| 6.9771        | 0.2352 | 2000 | 6.9348          | 0.0442   |
| 6.9665        | 0.2588 | 2200 | 6.9311          | 0.0465   |
| 6.9705        | 0.2823 | 2400 | 6.9251          | 0.0455   |
| 6.9432        | 0.3058 | 2600 | 6.9230          | 0.0443   |
| 6.9481        | 0.3293 | 2800 | 6.9079          | 0.0462   |
| 6.9149        | 0.3529 | 3000 | 6.8942          | 0.0466   |
| 6.9246        | 0.3764 | 3200 | 6.8954          | 0.0465   |
| 6.9335        | 0.3999 | 3400 | 6.8837          | 0.0461   |
| 6.9111        | 0.4234 | 3600 | 6.8785          | 0.0469   |
| 6.906         | 0.4470 | 3800 | 6.8752          | 0.0469   |
| 6.9177        | 0.4705 | 4000 | 6.8713          | 0.0471   |
| 6.911         | 0.4940 | 4200 | 6.8626          | 0.0465   |
| 6.8861        | 0.5175 | 4400 | 6.8574          | 0.0477   |
| 6.8846        | 0.5410 | 4600 | 6.8512          | 0.0479   |
| 6.9018        | 0.5646 | 4800 | 6.8519          | 0.0479   |
| 6.8692        | 0.5881 | 5000 | 6.8212          | 0.0497   |
| 5.3156        | 0.6116 | 5200 | 5.1641          | 0.1907   |
| 4.7809        | 0.6351 | 5400 | 4.7044          | 0.2167   |
| 4.5233        | 0.6587 | 5600 | 4.4978          | 0.2293   |
| 4.3777        | 0.6822 | 5800 | 4.3069          | 0.2636   |
| 4.2335        | 0.7057 | 6000 | 4.1444          | 0.2900   |
| 4.0578        | 0.7292 | 6200 | 3.9855          | 0.3337   |
| 3.9136        | 0.7528 | 6400 | 3.8571          | 0.3514   |
| 3.7782        | 0.7763 | 6600 | 3.7425          | 0.3723   |
| 3.6984        | 0.7998 | 6800 | 3.6417          | 0.3878   |
| 3.6353        | 0.8233 | 7000 | 3.5697          | 0.4025   |
| 3.5529        | 0.8469 | 7200 | 3.5094          | 0.4072   |
| 3.5186        | 0.8704 | 7400 | 3.4624          | 0.4116   |
| 3.4815        | 0.8939 | 7600 | 3.4289          | 0.4163   |
| 3.4733        | 0.9174 | 7800 | 3.4066          | 0.4212   |
| 3.4416        | 0.9410 | 8000 | 3.3904          | 0.4223   |
| 3.4265        | 0.9645 | 8200 | 3.3843          | 0.4215   |
| 3.4045        | 0.9880 | 8400 | 3.3817          | 0.4227   |


### Framework versions

- Transformers 4.47.1
- Pytorch 2.5.1
- Datasets 3.5.0
- Tokenizers 0.21.1
