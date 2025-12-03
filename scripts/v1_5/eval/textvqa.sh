#!/bin/bash

model_name=llava-fusefistv3-modifiedv4
model_path=/group/30106/aidenqian/TokenPacker-main/checkpoints/llava-fusefirstv3-modified-v4-finetune

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_loader \
    --model-path ${model_path} \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /group/30106/aidenqian/prismatic-vlms-main/data/download/llava-v1.5-instruct/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/${model_name}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${model_name}.jsonl
