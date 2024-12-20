import json
import os
import re
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from typing import Any, Tuple
from math import exp
from collections import defaultdict
from typing import List, Dict

from utils.constants import PROMPT_FORMATS


def performance_data_to_df(result_path, args, num_runs) -> pd.DataFrame:
    performance_data = read_performance_data_f1_prec_rec(
        output_dir=result_path,
        dataset_name=args.dataset,
        prompt_types=[
            "natural",
            "tabular",
            "json",
            "yaml",
            "narayan",
            "peeters",
        ],
        k=args.k,
        # num_runs=MANHEIM_DATASET_SIZES[args.dataset],
        num_runs=num_runs,
    )

    # print(performance_data.head())
    performance_data = performance_data.sort_values("gpt version")

    return performance_data


def softmax(logprobs):
    exp_probs = {label: exp(lp) for label, lp in logprobs.items()}
    total = sum(exp_probs.values())
    return {label: prob / total for label, prob in exp_probs.items()}


def read_performance_data_f1_prec_rec(
    output_dir: str = "output",
    dataset_name: str = None,
    prompt_types: List[str] = PROMPT_FORMATS,
    k: int = 0,
    num_runs: int = 200,
) -> pd.DataFrame:
    aggregated_data = agg_data_from_files(
        output_dir, dataset_name, prompt_types, k, num_runs
    )
    # print("Aggregated data")
    # print(aggregated_data)

    return prepare_dataframe(aggregated_data)


def compute_self_consistent_f1(responses: Dict[str, List[Tuple[str, str]]]) -> float:
    predictions = []
    actuals = []

    for prompt, pairs in responses.items():
        response_count = defaultdict(int)
        actual_labels = set()

        for pred, actual in pairs:
            response_count[pred] += 1
            if actual:  # Ensure actual label is not empty
                actual_labels.add(actual)

        if not actual_labels:
            continue

        most_popular_response = max(response_count, key=response_count.get)
        actual_label = next(
            iter(actual_labels)
        )  # Assuming consistent actual labels per prompt

        predictions.append(most_popular_response)
        actuals.append(actual_label)

    return (
        f1_score(actuals, predictions, pos_label="yes")
        if actuals and predictions
        else 0.0
    )


def prepare_dataframe(aggregated_data: Dict[Tuple[str, str, str], Any]) -> pd.DataFrame:
    data_for_df = []

    for key, value in aggregated_data.items():
        f1_sc = compute_self_consistent_f1(value["responses"])

        file_count = value["file_count"]

        data_for_df.append(
            {
                "gpt version": key[0],
                "prompt type": key[1],
                "dataset": key[2],
                "f1": np.mean(value["f1"]),
                "prec": np.mean(value["prec"]),
                "rec": np.mean(value["rec"]),
                "f1_std": np.std(value["f1"]),
                "prec_std": np.std(value["prec"]),
                "rec_std": np.std(value["rec"]),
                "f1_sc": f1_sc,
                "tp": value["tp"] / file_count,
                "tn": value["tn"] / file_count,
                "fp": value["fp"] / file_count,
                "fn": value["fn"] / file_count,
                "avg_tp_confidence": value["avg_tp_confidence"],
                "avg_tn_confidence": value["avg_tn_confidence"],
                "avg_fp_confidence": value["avg_fp_confidence"],
                "avg_fn_confidence": value["avg_fn_confidence"],
            }
        )

    return pd.DataFrame(data_for_df)


def agg_data_from_files(
    output_dir: str,
    dataset_name: str,
    prompt_types: List[str],
    k: int,
    num_runs: int = 200,
) -> Dict:
    aggregated_data = {}
    for root, dirs, files in os.walk(output_dir):
        current_dataset = os.path.basename(root)
        if dataset_name and current_dataset != dataset_name:
            continue

        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                with open(path, "r") as f:
                    metrics = json.load(f)

                    match = re.match(
                        rf"(gpt-\d+[a-z]?)_k_{k}_n_{num_runs}_(.+)_(\d{{2}})\.json",
                        file,
                    )

                    if match:
                        gpt_version = match.group(1)
                        prompt_type = match.group(2)
                        if prompt_type not in prompt_types:
                            continue

                        key = (gpt_version, prompt_type, dataset_name)

                        if key not in aggregated_data:
                            aggregated_data[key] = {
                                "f1": [],
                                "prec": [],
                                "rec": [],
                                "responses": defaultdict(list),
                                "tp": 0,
                                "tn": 0,
                                "fp": 0,
                                "fn": 0,
                                "file_count": 0,  # Counter for the number of files processed
                                "tp_confidence": [],  # Confidence for True Positives
                                "tn_confidence": [],  # Confidence for True Negatives
                                "fp_confidence": [],  # Confidence for False Positives
                                "fn_confidence": [],  # Confidence for False Negatives
                            }

                        aggregated_data[key]["file_count"] += 1
                        # Append traditional metrics
                        aggregated_data[key]["f1"].append(metrics.get("f1", [0])[0])
                        aggregated_data[key]["prec"].append(metrics.get("prec", [0])[0])
                        aggregated_data[key]["rec"].append(metrics.get("rec", [0])[0])

                        # Collect responses for self-consistency
                        for detail in metrics.get("details", []):
                            prompt = detail["prompt"]
                            predicted_label = detail["predicted_label"]
                            actual_label = detail["actual_label"]
                            logprobs = detail.get("logprob", {})

                            # Calculate softmax probabilities
                            softmax_probs = softmax(logprobs)

                            # Get the confidence for the predicted label
                            confidence = softmax_probs.get(
                                predicted_label.capitalize(), 0
                            )

                            aggregated_data[key]["responses"][prompt].append(
                                (predicted_label, actual_label)
                            )

                            # Update counts and track confidence
                            if predicted_label == "yes" and actual_label == "yes":
                                aggregated_data[key]["tp"] += 1
                                aggregated_data[key]["tp_confidence"].append(confidence)
                            elif predicted_label == "no" and actual_label == "no":
                                aggregated_data[key]["tn"] += 1
                                aggregated_data[key]["tn_confidence"].append(confidence)
                            elif predicted_label == "yes" and actual_label == "no":
                                aggregated_data[key]["fp"] += 1
                                aggregated_data[key]["fp_confidence"].append(confidence)
                            elif predicted_label == "no" and actual_label == "yes":
                                aggregated_data[key]["fn"] += 1
                                aggregated_data[key]["fn_confidence"].append(confidence)

    # After aggregation, compute average confidences for each key
    for key, data in aggregated_data.items():
        data["avg_tp_confidence"] = (
            sum(data["tp_confidence"]) / len(data["tp_confidence"])
            if data["tp_confidence"]
            else 0
        )
        data["avg_tn_confidence"] = (
            sum(data["tn_confidence"]) / len(data["tn_confidence"])
            if data["tn_confidence"]
            else 0
        )
        data["avg_fp_confidence"] = (
            sum(data["fp_confidence"]) / len(data["fp_confidence"])
            if data["fp_confidence"]
            else 0
        )
        data["avg_fn_confidence"] = (
            sum(data["fn_confidence"]) / len(data["fn_confidence"])
            if data["fn_confidence"]
            else 0
        )

    # print(aggregated_data["gpt-4o", "json", "dblp_gs"]["avg_fp_confidence"])
    return aggregated_data


def read_performance_categories():
    # Directory structure
    base_dir = "output/output_basic_ctx"
    datasets = [
        "abt-buy",
        "amazon",
        "dblp_acm",
        "dblp_acm_dirty",
        "dblp_gs",
        "dblp_gs_dirty",
    ]
    pn_ratios = ["0_100", "30_70", "50_50", "70_30", "100_0"]
    models = ["gpt-3", "gpt-4", "gpt-4o"]

    # Prepare list to hold data
    data = []

    # Loop through directories and files
    for dataset in datasets:
        for pn_ratio in pn_ratios:
            folder_path = os.path.join(base_dir, dataset, "P_N", pn_ratio)
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(".json"):
                        model = file.split("_")[0]
                        file_path = os.path.join(folder_path, file)
                        with open(file_path, "r") as f:
                            results = json.load(f)
                            # Extract metrics
                            prec = results.get("prec", [None])[0]
                            rec = results.get("rec", [None])[0]
                            f1 = results.get("f1", [None])[0]
                            acc = results.get("acc", [None])[0]
                            time = results.get("time", [None])[0]

                            # Extract TP, FP, FN, TN counts
                            details = results.get("details", [])
                            tp = sum(
                                1
                                for d in details
                                if d["predicted_label"] == "yes"
                                and d["actual_label"] == "yes"
                            )
                            fp = sum(
                                1
                                for d in details
                                if d["predicted_label"] == "yes"
                                and d["actual_label"] == "no"
                            )
                            fn = sum(
                                1
                                for d in details
                                if d["predicted_label"] == "no"
                                and d["actual_label"] == "yes"
                            )
                            tn = sum(
                                1
                                for d in details
                                if d["predicted_label"] == "no"
                                and d["actual_label"] == "no"
                            )

                            # Append data
                            data.append(
                                {
                                    "Dataset": dataset,
                                    "PN_Ratio": pn_ratio,
                                    "Model": model,
                                    "Precision": prec,
                                    "Recall": rec,
                                    "F1": f1,
                                    "Accuracy": acc,
                                    "Time": time,
                                    "TP": tp,
                                    "FP": fp,
                                    "FN": fn,
                                    "TN": tn,
                                }
                            )

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Multiply numbers (except (TN, TP, FN, FP) by 100 and round to 2 decimals
    df[["Precision", "Recall", "F1", "Accuracy"]] = (
        df[["Precision", "Recall", "F1", "Accuracy"]].apply(lambda x: x * 100).round(2)
    )

    return df
