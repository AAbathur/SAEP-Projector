#!/bin/bash

#model_path=checkpoints/llava-tokenpacker-910b-v2
#model_path=checkpoints/llava-fusefirstv3-modified-v4-finetune
NAME=llava-13b-fusefisrtv3modifiedv4
model_path=checkpoints/llava-13b-fusefirstv3-modified-v4-finetune



#NAME=llava-fusefisrtv3modifiedv4
#model_path=/group/30106/aidenqian/TokenPacker-main/checkpoints/llava-fusefirstv3-modified-v4-finetune

#python -m llava.eval.model_vqa_loader \
#    --model-path $model_path\
#    --question-file /group/30106/aidenqian/TokenPacker-main/playground/data/eval/vizwiz/llava_test.jsonl \
#    --image-folder /group/30106/aidenqian/prismatic-vlms-main/vlm-evaluation/download/vizwiz/test_images \
#    --answers-file ./playground/data/eval/vizwiz/answers/${NAME}.jsonl \
#    --temperature 0 \
#    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/${NAME}.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/${NAME}.json
