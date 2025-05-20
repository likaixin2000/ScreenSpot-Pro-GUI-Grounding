#!/bin/bash
set -e

# English

models=("cogagent24" "ariaui" "uground" "osatlas-7b" "osatlas-4b" "showui" "seeclick" "qwen1vl" "qwen2vl" "minicpmv" "cogagent" "gpt4o" )
for model in "${models[@]}"
do
    python eval_screenspot_pro.py  \
        --model_type ${model}  \
        --screenspot_imgs "../data/ScreenSpot-Pro/images"  \
        --screenspot_test "../data/ScreenSpot-Pro/annotations"  \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "./results/${model}.json" \
        --inst_style "instruction"

done

# Qwen 2.5-VL series
ckpts=("Qwen/Qwen2.5-VL-3B-Instruct" "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen/Qwen2.5-VL-72B-Instruct")
for ckpt in "${ckpts[@]}"
do
    python eval_screenspot_pro_parallel.py  \
        --model_type "qwen2_5vl"  \
        --model_name_or_path ${ckpt}  \
        --screenspot_imgs "../data/ScreenSpot-Pro/images"  \
        --screenspot_test "../data/ScreenSpot-Pro/annotations"  \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "./results/${ckpt}.json" \
        --inst_style "instruction"
done
