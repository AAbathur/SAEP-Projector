#!/bin/bash
#NAME=llava-tokenpacker-7b
model_name=llava-fusefistv3-modifiedv4
model_path=/group/30106/aidenqian/TokenPacker-main/checkpoints/llava-fusefirstv3-modified-v4-finetune

img_dir=/group/30106/aidenqian/VLMEvalKit/eval_data/images/MME

#python -m llava.eval.model_vqa_loader \
#    --model-path ${model_path} \
#    --question-file ./playground/data/eval/MME/llava_mme-our.jsonl \
#    --image-folder ${img_dir} \
#    --answers-file ./playground/data/eval/MME/answers/${model_name}.jsonl \
#    --temperature 0 \
#    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $model_name

#cd eval_tool

#python calculation.py --results_dir answers/$NAME

