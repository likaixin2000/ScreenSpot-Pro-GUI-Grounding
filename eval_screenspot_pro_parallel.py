# Key notes:
# - Don't use transformers' device_map="auto" as it spreads the model across all available GPUs.
# - Use multiprocessing with 'spawn' method to avoid issues with CUDA in worker processes, especially with vLLM.

import copy
import itertools
import torch
import json
import re
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
import multiprocessing as mp
mp.set_start_method('spawn', force=True)  # Use 'spawn' to avoid issues with CUDA in multiprocessing

from model_factory import build_model

logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)

GT_TYPES = ['positive', 'negative']
INSTRUCTION_STYLES = ['instruction', 'action', 'description']
LANGUAGES = ['en', 'cn']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=False)
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--inst_style', type=str, required=True, choices=INSTRUCTION_STYLES + ['all'], help="Instruction style to use.")
    parser.add_argument('--language', type=str, required=True, choices=LANGUAGES + ['all'], default='en', help="Language to use.")
    parser.add_argument('--gt_type', type=str, required=True, choices=GT_TYPES + ['all'], help="Ground truth type: 'positive' or 'negative'.")
    parser.add_argument('--log_path', type=str, required=True)

    parser.add_argument('--num_gpu', type=int, default=8, help="Number of GPUs to use for parallel processing.")
    args = parser.parse_args()
    return args


def initializer(gpu_queue, args):
    """Initialize each worker process with a unique GPU and load the model."""
    gpu_index = gpu_queue.get()
    print(f"Worker assigned to GPU {gpu_index}")
    torch.cuda.set_device(gpu_index)
    model = build_model(args)
    global worker_model
    worker_model = model

def worker_function(sample):
    """Process a single task on the assigned GPU."""
    global worker_model
    img_path = os.path.join(sample["screenspot_imgs"], sample["img_filename"])
    if sample["gt_type"] == "positive":
        response = worker_model.ground_only_positive(instruction=sample["prompt_to_evaluate"], image=img_path)
    elif sample["gt_type"] == "negative":
        response = worker_model.ground_allow_negative(instruction=sample["prompt_to_evaluate"], image=img_path)
    point = response["point"]
    img_size = sample["img_size"]
    point_in_pixel = [point[0] * img_size[0], point[1] * img_size[1]] if point else None
    sample_result = {
        "img_path": img_path,
        "group": sample["group"] if "group" in sample else None,
        "platform": sample["platform"],
        "application": sample["application"],
        "lang": sample["language"],
        "instruction_style": sample["instruction_style"],
        "prompt_to_evaluate": sample["prompt_to_evaluate"],
        "gt_type": sample["gt_type"],
        "ui_type": sample["ui_type"],
        "task_filename": sample["task_filename"],
        "pred": point_in_pixel,
        "raw_response": response["raw_response"]
    }
    if sample["gt_type"] == "positive":
        correctness = eval_sample_positive_gt(sample, response)
        sample_result.update({"bbox": sample["bbox"]})
    elif sample["gt_type"] == "negative":
        correctness = eval_sample_negative_gt(sample, response)
    sample_result.update({"correctness": correctness})
    return sample_result

def collect_results_to_eval(results, platform=None, group=None, application=None, language=None, gt_type=None, instruction_style=None, ui_type=None):
    # ... (unchanged from original)
    filtered_results = []
    for sample in results:
        if (platform is None or sample.get("platform") == platform) and \
           (group is None or sample.get("group") == group) and \
           (application is None or sample.get("application") == application) and \
           (language is None or sample.get("language") == language) and \
           (gt_type is None or sample.get("gt_type") == gt_type) and \
           (instruction_style is None or sample.get("instruction_style") == instruction_style) and \
           (ui_type is None or sample.get("ui_type") == ui_type):
            filtered_results.append(sample)
    return filtered_results

def make_combinations(results, platform=False, group=None, application=False, language=False, gt_type=False, instruction_style=False, ui_type=False):
    # ... (unchanged from original)
    unique_values = {
        "platform": set(),
        "group": set(),
        "application": set(),
        "language": set(),
        "gt_type": set(),
        "instruction_style": set(),
        "ui_type": set(),
    }
    for sample in results:
        if platform:
            unique_values["platform"].add(sample.get("platform"))
        if group:
            unique_values["group"].add(sample.get("group"))
        if application:
            unique_values["application"].add(sample.get("application"))
        if language:
            unique_values["language"].add(sample.get("language"))
        if gt_type:
            unique_values["gt_type"].add(sample.get("gt_type"))
        if instruction_style:
            unique_values["instruction_style"].add(sample.get("instruction_style"))
        if ui_type:
            unique_values["ui_type"].add(sample.get("ui_type"))
    filtered_values = {key: list(value) for key, value in unique_values.items() if value}
    if not filtered_values:
        return []
    attribute_combinations = list(itertools.product(*filtered_values.values()))
    combinations = [dict(zip(filtered_values.keys(), combination)) for combination in attribute_combinations]
    return combinations

def calc_metric_for_result_list(results):
    # ... (unchanged from original)
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")
    text_results = collect_results_to_eval(results, ui_type="text")
    icon_results = collect_results_to_eval(results, ui_type="icon")
    text_correct = sum(1 for res in text_results if res["correctness"] == "correct")
    text_total = len(text_results)
    icon_correct = sum(1 for res in icon_results if res["correctness"] == "correct")
    icon_total = len(icon_results)
    metrics = {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0,
        "text_acc": text_correct / text_total if text_total > 0 else 0,
        "icon_acc": icon_correct / icon_total if icon_total > 0 else 0
    }
    return metrics

def eval_sample_positive_gt(sample, response):
    # ... (unchanged from original)
    bbox = sample["bbox"]
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
    img_size = sample["img_size"]
    bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
    click_point = response["point"]
    print(click_point)
    if click_point is None:
        return "wrong_format"
    if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
        return "correct"
    else:
        return "wrong"

def eval_sample_negative_gt(sample, response):
    # ... (unchanged from original)
    if response["result"] == "negative":
        return "correct"
    elif response["result"] == "positive":
        return "wrong"
    else:
        return "wrong_format"

def evaluate_fine_grained(results):
    # ... (unchanged from original)
    combinations = make_combinations(results, platform=True, application=True, instruction_style=True, gt_type=True)
    evaluation_result = {}
    for combo in combinations:
        platform = combo.get("platform")
        application = combo.get("application")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        filtered_results = collect_results_to_eval(results, platform=platform, application=application, instruction_style=inst_style, gt_type=gt_type)
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        key = f"plat:{platform} app:{application} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics
    return evaluation_result

def evaluate_seeclick_paper_style(results):
    # ... (unchanged from original)
    combinations = make_combinations(results, platform=True, instruction_style=True, gt_type=True)
    evaluation_result = {}
    for combo in combinations:
        platform = combo.get("platform")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        filtered_results = collect_results_to_eval(results, platform=platform, instruction_style=inst_style, gt_type=gt_type)
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        key = f"plat:{platform} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics
    return evaluation_result

def evaluate_leaderboard_detailed_style(results):
    # ... (unchanged from original)
    combinations = make_combinations(results, application=True)
    evaluation_result = {}
    for combo in combinations:
        application = combo.get("application")
        filtered_results = collect_results_to_eval(results, application=application)
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        key = f"app:{application}"
        evaluation_result[key] = metrics
    return evaluation_result

def evaluate_leaderboard_simple_style(results):
    # ... (unchanged from original)
    combinations = make_combinations(results, group=True)
    evaluation_result = {}
    for combo in combinations:
        group = combo.get("group")
        filtered_results = collect_results_to_eval(results, group=group)
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        key = f"group:{group}"
        evaluation_result[key] = metrics
    return evaluation_result

def evaluate_overall(results):
    # ... (unchanged from original)
    metrics = calc_metric_for_result_list(results)
    return metrics

def evaluate(results):
    # ... (unchanged from original)
    result_report = {"details": [], "metrics": {}}
    result_report["metrics"]["fine_grained"] = evaluate_fine_grained(results)
    result_report["metrics"]["seeclick_style"] = evaluate_seeclick_paper_style(results)
    result_report["metrics"]["leaderboard_simple_style"] = evaluate_leaderboard_simple_style(results)
    result_report["metrics"]["leaderboard_detailed_style"] = evaluate_leaderboard_detailed_style(results)
    result_report["metrics"]["overall"] = evaluate_overall(results)
    result_report["details"] = results
    return result_report

def main(args):
    if args.task == "all":
        task_filenames = [os.path.splitext(f)[0] for f in os.listdir(args.screenspot_test) if f.endswith(".json")]
    else:
        task_filenames = args.task.split(",")

    if args.inst_style == "all":
        inst_styles = INSTRUCTION_STYLES
    else:
        inst_styles = args.inst_style.split(",")

    if args.language == "all":
        languages = LANGUAGES
    else:
        languages = args.language.split(",")

    if args.gt_type == "all":
        gt_types = GT_TYPES
    else:
        gt_types = args.gt_type.split(",")

    tasks_to_run = []
    for task_filename in task_filenames:
        dataset = task_filename + ".json"
        with open(os.path.join(args.screenspot_test, dataset), 'r') as f:
            task_data = json.load(f)
        for inst_style in inst_styles:
            for gt_type in gt_types:
                for lang in languages:
                    for task_instance in task_data:
                        task_instance = copy.deepcopy(task_instance)
                        task_instance["task_filename"] = task_filename
                        task_instance["gt_type"] = gt_type
                        task_instance["instruction_style"] = inst_style
                        task_instance["language"] = lang
                        if lang == "cn":
                            if inst_style != 'instruction' or gt_type != 'positive':
                                raise AttributeError("Only positive samples and 'instruction' style are supported for Chinese instructions.")
                            task_instance["prompt_to_evaluate"] = task_instance["instruction_cn"]
                        elif lang == "en":
                            task_instance["prompt_to_evaluate"] = task_instance["instruction"]
                        task_instance["screenspot_imgs"] = args.screenspot_imgs  # Add screenspot_imgs to each task
                        tasks_to_run.append(task_instance)
        print(f"Num of sample in {task_filename}: {len(task_data)} * {len(inst_styles)} * {len(gt_types)} * {len(languages)} = {len(task_data) * len(inst_styles) * len(gt_types) * len(languages)}")
    print(f"Total tasks: {len(tasks_to_run)}")

    # Create GPU queue for assigning GPUs to workers
    gpu_queue = mp.Queue()
    for i in range(args.num_gpu):
        gpu_queue.put(i)

    # Process tasks in parallel across 8 GPUs
    with mp.Pool(processes=args.num_gpu, initializer=initializer, initargs=(gpu_queue, args)) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(worker_function, tasks_to_run), total=len(tasks_to_run)):
            results.append(result)

    # Evaluate and save results
    result_report = evaluate(results)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, 'w') as f:
        json.dump(result_report, f, indent=4)
    logging.info("Evaluation of ScreenSpot finished.")

if __name__ == "__main__":
    main(parse_args())
