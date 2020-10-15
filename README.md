# Length-Adaptive Transformer

This is the official Pytorch implementation of [**Length-Adaptive Transformer**](https://arxiv.org/abs/2010.07003).
For detailed information about the method, please refer to our paper.

Our code is based on [HuggingFace's (🤗) Transformers](https://github.com/huggingface/transformers) library.
Currently, it only supports limited transformers (BERT and DistilBERT) and downstream tasks (SQuAD 1.1 and GLUE benchmark).
We will extend it one-by-one to support other transformers and tasks.
You can easily apply our method to any other use cases beforehand. 


## Getting Started


### Requirements
- Python 3
- PyTorch
- 🤗 Transformers
- torchprofile (to measure FLOPs)


### Dataset Preparation
- SQuAD 1.1: Downoad following files in a `$SQUAD_DIR` directory:
[train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json), [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json), and [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py).
- GLUE: Run GLUE data [download script](https://github.com/nyu-mll/jiant/blob/master/scripts/download_glue_data.py) to download data in a `$GLUE_DIR` directory. 


### (Standard) Finetuning pretrained transformer
For SQuAD 1.1, use `run_squad.py` slightly modified from 🤗 Transformers' [question-answering example](https://github.com/huggingface/transformers/tree/master/examples/question-answering).
```
python run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --save_only_best \
  --do_lower_case \
  --data_dir $SQUAD_DIR \
  --train_file train-v1.1.json \
  --predict_file dev-v1.1.json \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $SQUAD_OUTPUT_DIR/standard
```

For GLUE, use `run_glue.py` slightly modified from 🤗 Transformers' [text-classification example](https://github.com/huggingface/transformers/tree/master/examples/text-classification).
```
python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir $GLUE_OUTPUT_DIR/$TASK_NAME/standard
```


### Training with LengthDrop
Starting from a checkpoint finetuned without *Drop-and-Restore*, continue finetuning for additional steps with *Drop-and-Restore* and *LengthDrop*.

```
python run_squad.py \
  --model_type bert \
  --model_name_or_path $SQUAD_OUTPUT_DIR/standard/checkpoint-best \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --save_only_best \
  --do_lower_case \
  --data_dir $SQUAD_DIR \
  --train_file train-v1.1.json \
  --predict_file dev-v1.1.json \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 5.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $SQUAD_OUTPUT_DIR/length_adaptive \
  --length_adaptive \
  --num_sandwich 2 \
  --length_drop_ratio_bound 0.2 \
  --layer_dropout_prob 0.2 \
```

```
python run_glue.py \
  --model_name_or_path $GLUE_OUTPUT_DIR/$TASK_NAME/standard/checkpoint-best \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir $GLUE_OUTPUT_DIR/$TASK_NAME/length_adaptive
  --length_adaptive \
  --num_sandwich 2 \
  --length_drop_ratio_bound 0.2 \
  --layer_dropout_prob 0.2 \
```


### Evolutionary Search of Length Configurations
After training with *LengthDrop*, perform an evolutionary search to find length configurations for anytime prediction.

```
python run_squad.py \
  --model_type bert \
  --model_name_or_path $SQUAD_OUTPUT_DIR/length_adaptive/checkpoint-best \
  --do_search \
  --do_lower_case \
  --data_dir $SQUAD_DIR \
  --train_file train-v1.1.json \
  --predict_file dev-v1.1.json \
  --per_gpu_eval_batch_size 32 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $SQUAD_OUTPUT_DIR/evolutionary_search \
  --evo_iter 30 \
  --mutation_size 30 \
  --crossover_size 30 \
```

```
python run_glue.py \
  --model_name_or_path $GLUE_OUTPUT_DIR/$TASK_NAME/length_adaptive/checkpoint-best \
  --task_name $TASK_NAME \
  --do_search \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_eval_batch_size 32 \
  --output_dir $GLUE_OUTPUT_DIR/$TASK_NAME/evolutionary_search
  --evo_iter 30 \
  --mutation_size 30 \
  --crossover_size 30 \
```


## License

```
Copyright 2020-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

