---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:612
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- accuracy
- accuracy_threshold
- f1
- f1_threshold
- precision
- recall
- average_precision
model-index:
- name: CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2
  results:
  - task:
      type: cross-encoder-binary-classification
      name: Cross Encoder Binary Classification
    dataset:
      name: medical reranker
      type: medical-reranker
    metrics:
    - type: accuracy
      value: 0.9259259259259259
      name: Accuracy
    - type: accuracy_threshold
      value: -1.1257927417755127
      name: Accuracy Threshold
    - type: f1
      value: 0.9344262295081968
      name: F1
    - type: f1_threshold
      value: -1.325373649597168
      name: F1 Threshold
    - type: precision
      value: 0.9047619047619048
      name: Precision
    - type: recall
      value: 0.9661016949152542
      name: Recall
    - type: average_precision
      value: 0.9676510876506522
      name: Average Precision
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'BertForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['Does health information exchange reduce redundant imaging?', 'the same study in the same body region performed within 30 days at unaffiliated EDs. In our sample there were 20,139 repeat CTs (representing 14.7% of those cases with CT in the index visit), 13,060 repeat ultrasounds (20.7% of ultrasound cases), and 29,703 repeat chest x-rays (19.5% of x-ray cases). HIE was associated with reduced probability of repeat ED imaging in all 3 modalities: -8.7 percentage points for CT [95% confidence interval (CI): -14.7, -2.7], -9.1 percentage points for ultrasound (95% CI: -17.2, -1.1), and -13.0 percentage points for chest x-ray (95% CI: -18.3, -7.7), reflecting reductions of 44%-67% relative'],
    ['Epidural analgesia for surgical treatment of peritoneal carcinomatosis: a risky technique?', 'subjects (34%), and after HIPEC in 23 subjects (66%). The median dose of ropivacain given peroperatively in the epidural catheter was 40 mg (30-75). Norepinephrin was used in two subjects (6%) peroperatively (median infusion rate 0.325 μg/kg per minute [0.32-0.33]), and in four subjects (11%) in the postoperative 24 hours. No spinal haematoma, meningitis or epidural abscess were noted. Five subjects (14%) had a thrombopenia or a prothrombin time less than 60% before catheter removal. Two subjects (6%) had a leukopenia before catheter removal. No thrombopenia or blood coagulation disorders were recorded the day of catheter removal.'],
    ['Does sex affect the outcome of laparoscopic cholecystectomy?', '≤ 0.05 was considered significant. The study included 1772 female and 289 male patients. The mean age for male patients was 44.07 ± 11.91 years compared to 41.29 ± 12.18 years for female patients (P = 0.706). Laparoscopic cholecystectomy was successfully completed in 1996 patients. The conversion rate was higher in men (P < 0.001), and the mean operating time was longer in men (P < 0.001). Bile duct injuries occurred more frequently in men (P < 0.001). Gallbladder perforation and gallstone spillage also occurred more commonly in men (P = 0.001); similarly severe inflammation was reported more in male'],
    ['Is fetal gender associated with emergency department visits for asthma during pregnancy?', 'To investigate if fetal gender (1) affects the risk of having an emergency department (ED) visit for asthma; and (2) is associated with adverse pregnancy outcomes among women who had at least one visit to the ED for asthma during pregnancy. We linked two provincial administrative databases containing records on in-patient deliveries and ED visits. The study sample included women who delivered a live singleton baby between April 2003 and March 2004. Pregnant women who made at least one ED visit for asthma were counted as cases and the rest of the women as control subjects. We performed a multivariable'],
    ['What is the homocysteine level and is it elevated?', 'Immunoassay Test Result Unit Biological Ref. Interval Vitamin B12 L < 148 pg/mL 187 - 833 CLIA Vitamin B12 is essential in DNA synthesis, hematopoiesis, and CNS integrity. Interpretation: • Increased In : Chronic granulocytic leukemia, COPD and Chronic renal failure, Leukocytosis, Liver cell damage (hepatitis, cirrhosis), Obesity and Severe CHF, Polycythemia vera, Protein malnutrition. • Decreased In : Abnormalities of cobalamin transport or metabolism, Bacterial overgrowth, Crohn disease, Dietary deficiency (e.g. in vegetarians), Diphyllobothrium (fish tapeworm) infestation, Gastric or small intestine surgery, Hypochlorhydria, Inflammatory bowel disease, Intestinal malabsorption and Intrinsic factor deficiency Limitations: • Drugs such as chloral hydrate increase vitamin B12 levels. On the other hand, alcohol, aminosalicylic acid, anticonvulsants, ascorbic acid, cholestyramine, cimetidine, colchicines, metformin, neomycin, oral contraceptives, ranitidine, and triamterene decrease vitamin B12 levels. • The evaluation of macrocytic anemia requires measurements of both vitamin B12 and folate levels; ideally they should be measured simultaneously. • Specimen collection soon after blood transfusion can falsely increase vitamin B12 levels. • Patients taking vitamin B12 supplementation may have misleading results. • A normal serum concentration of B12 does not rule out tissue deficiency of vitamin B12. The most sensitive test for B12 deficiency at the cellular level is'],
]
scores = model.predict(pairs)
print(scores)
# [ 2.456   1.9888  5.4154 10.5833 -1.3449]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Does health information exchange reduce redundant imaging?',
    [
        'the same study in the same body region performed within 30 days at unaffiliated EDs. In our sample there were 20,139 repeat CTs (representing 14.7% of those cases with CT in the index visit), 13,060 repeat ultrasounds (20.7% of ultrasound cases), and 29,703 repeat chest x-rays (19.5% of x-ray cases). HIE was associated with reduced probability of repeat ED imaging in all 3 modalities: -8.7 percentage points for CT [95% confidence interval (CI): -14.7, -2.7], -9.1 percentage points for ultrasound (95% CI: -17.2, -1.1), and -13.0 percentage points for chest x-ray (95% CI: -18.3, -7.7), reflecting reductions of 44%-67% relative',
        'subjects (34%), and after HIPEC in 23 subjects (66%). The median dose of ropivacain given peroperatively in the epidural catheter was 40 mg (30-75). Norepinephrin was used in two subjects (6%) peroperatively (median infusion rate 0.325 μg/kg per minute [0.32-0.33]), and in four subjects (11%) in the postoperative 24 hours. No spinal haematoma, meningitis or epidural abscess were noted. Five subjects (14%) had a thrombopenia or a prothrombin time less than 60% before catheter removal. Two subjects (6%) had a leukopenia before catheter removal. No thrombopenia or blood coagulation disorders were recorded the day of catheter removal.',
        '≤ 0.05 was considered significant. The study included 1772 female and 289 male patients. The mean age for male patients was 44.07 ± 11.91 years compared to 41.29 ± 12.18 years for female patients (P = 0.706). Laparoscopic cholecystectomy was successfully completed in 1996 patients. The conversion rate was higher in men (P < 0.001), and the mean operating time was longer in men (P < 0.001). Bile duct injuries occurred more frequently in men (P < 0.001). Gallbladder perforation and gallstone spillage also occurred more commonly in men (P = 0.001); similarly severe inflammation was reported more in male',
        'To investigate if fetal gender (1) affects the risk of having an emergency department (ED) visit for asthma; and (2) is associated with adverse pregnancy outcomes among women who had at least one visit to the ED for asthma during pregnancy. We linked two provincial administrative databases containing records on in-patient deliveries and ED visits. The study sample included women who delivered a live singleton baby between April 2003 and March 2004. Pregnant women who made at least one ED visit for asthma were counted as cases and the rest of the women as control subjects. We performed a multivariable',
        'Immunoassay Test Result Unit Biological Ref. Interval Vitamin B12 L < 148 pg/mL 187 - 833 CLIA Vitamin B12 is essential in DNA synthesis, hematopoiesis, and CNS integrity. Interpretation: • Increased In : Chronic granulocytic leukemia, COPD and Chronic renal failure, Leukocytosis, Liver cell damage (hepatitis, cirrhosis), Obesity and Severe CHF, Polycythemia vera, Protein malnutrition. • Decreased In : Abnormalities of cobalamin transport or metabolism, Bacterial overgrowth, Crohn disease, Dietary deficiency (e.g. in vegetarians), Diphyllobothrium (fish tapeworm) infestation, Gastric or small intestine surgery, Hypochlorhydria, Inflammatory bowel disease, Intestinal malabsorption and Intrinsic factor deficiency Limitations: • Drugs such as chloral hydrate increase vitamin B12 levels. On the other hand, alcohol, aminosalicylic acid, anticonvulsants, ascorbic acid, cholestyramine, cimetidine, colchicines, metformin, neomycin, oral contraceptives, ranitidine, and triamterene decrease vitamin B12 levels. • The evaluation of macrocytic anemia requires measurements of both vitamin B12 and folate levels; ideally they should be measured simultaneously. • Specimen collection soon after blood transfusion can falsely increase vitamin B12 levels. • Patients taking vitamin B12 supplementation may have misleading results. • A normal serum concentration of B12 does not rule out tissue deficiency of vitamin B12. The most sensitive test for B12 deficiency at the cellular level is',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Binary Classification

* Dataset: `medical-reranker`
* Evaluated with [<code>CEBinaryClassificationEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CEBinaryClassificationEvaluator)

| Metric                | Value      |
|:----------------------|:-----------|
| accuracy              | 0.9259     |
| accuracy_threshold    | -1.1258    |
| f1                    | 0.9344     |
| f1_threshold          | -1.3254    |
| precision             | 0.9048     |
| recall                | 0.9661     |
| **average_precision** | **0.9677** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 612 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 612 samples:
  |         | sentence_0                                                                         | sentence_1                                                                          | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              | float                                                          |
  | details | <ul><li>min: 10 tokens</li><li>mean: 19.86 tokens</li><li>max: 41 tokens</li></ul> | <ul><li>min: 10 tokens</li><li>mean: 193.1 tokens</li><li>max: 404 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.56</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                              | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | label            |
  |:--------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Does health information exchange reduce redundant imaging?</code>                                 | <code>the same study in the same body region performed within 30 days at unaffiliated EDs. In our sample there were 20,139 repeat CTs (representing 14.7% of those cases with CT in the index visit), 13,060 repeat ultrasounds (20.7% of ultrasound cases), and 29,703 repeat chest x-rays (19.5% of x-ray cases). HIE was associated with reduced probability of repeat ED imaging in all 3 modalities: -8.7 percentage points for CT [95% confidence interval (CI): -14.7, -2.7], -9.1 percentage points for ultrasound (95% CI: -17.2, -1.1), and -13.0 percentage points for chest x-ray (95% CI: -18.3, -7.7), reflecting reductions of 44%-67% relative</code>       | <code>1.0</code> |
  | <code>Epidural analgesia for surgical treatment of peritoneal carcinomatosis: a risky technique?</code> | <code>subjects (34%), and after HIPEC in 23 subjects (66%). The median dose of ropivacain given peroperatively in the epidural catheter was 40 mg (30-75). Norepinephrin was used in two subjects (6%) peroperatively (median infusion rate 0.325 μg/kg per minute [0.32-0.33]), and in four subjects (11%) in the postoperative 24 hours. No spinal haematoma, meningitis or epidural abscess were noted. Five subjects (14%) had a thrombopenia or a prothrombin time less than 60% before catheter removal. Two subjects (6%) had a leukopenia before catheter removal. No thrombopenia or blood coagulation disorders were recorded the day of catheter removal.</code> | <code>1.0</code> |
  | <code>Does sex affect the outcome of laparoscopic cholecystectomy?</code>                               | <code>≤ 0.05 was considered significant. The study included 1772 female and 289 male patients. The mean age for male patients was 44.07 ± 11.91 years compared to 41.29 ± 12.18 years for female patients (P = 0.706). Laparoscopic cholecystectomy was successfully completed in 1996 patients. The conversion rate was higher in men (P < 0.001), and the mean operating time was longer in men (P < 0.001). Bile duct injuries occurred more frequently in men (P < 0.001). Gallbladder perforation and gallstone spillage also occurred more commonly in men (P = 0.001); similarly severe inflammation was reported more in male</code>                                | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | medical-reranker_average_precision |
|:-----:|:----:|:----------------------------------:|
| 1.0   | 39   | 0.9672                             |
| 2.0   | 78   | 0.9681                             |
| 3.0   | 117  | 0.9677                             |


### Training Time
- **Training**: 13.8 minutes

### Framework Versions
- Python: 3.12.3
- Sentence Transformers: 5.4.1
- Transformers: 5.8.0
- PyTorch: 2.11.0+cpu
- Accelerate: 1.13.0
- Datasets: 4.8.5
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->