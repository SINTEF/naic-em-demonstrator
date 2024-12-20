from openai import OpenAI

import re
import pandas as pd
import logging
import json

from utils.constants import CATEGORY_ID, CATEGORY_PAIRS

client = OpenAI()


def get_completion_from_messages(
    messages,
    model="gpt-4-1106-preview",
    temperature=0,
    max_tokens=1000,
):
    response = client.chat.completions.create(
        model=model,
        seed=1234,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=2,
    )

    highest_logprob = response.choices[0].logprobs.content[0].top_logprobs[0]
    lowest_logprob = response.choices[0].logprobs.content[0].top_logprobs[1]

    logprob_dict = {
        highest_logprob.token: highest_logprob.logprob,
        lowest_logprob.token: lowest_logprob.logprob,
    }

    return response.choices[0].message.content, logprob_dict


def load_result_files(result_dir, dataset_name, model, k, num_pairs, format_, run_id):
    file_path = f"{result_dir}/{dataset_name}/{model}_k_{k}_n_{num_pairs}_{format_}_{run_id}.json"
    with open(file_path, "r") as file:
        data = json.load(file)
    return data["details"]


def get_examples(
    prompt_method,
    columns,
    dataset_type,
    K: int = 10,
    dataset: str = "amazon",
    category_pair: tuple = ("P", "N"),
    ratio: str = "50/50",
):
    if category_pair is None:
        category_pair = ("P", "N")

    # Parse the category pair
    left_category, right_category = category_pair

    # Parse the ratio
    left_ratio, right_ratio = map(int, ratio.split("/"))
    total_ratio = left_ratio + right_ratio

    # Calculate number of positive and negative examples
    num_left_category = (left_ratio * K) // total_ratio
    num_right_category = K - num_left_category

    logging.info(f"{CATEGORY_PAIRS[left_category]} examples: {num_left_category}")
    logging.info(f"{CATEGORY_PAIRS[right_category]} examples: {num_right_category}")

    left_examples_path = (
        f"category_selections/{dataset}/{dataset}_{CATEGORY_ID[left_category]}.csv"
    )
    right_examples_path = (
        f"category_selections/{dataset}/{dataset}_{CATEGORY_ID[right_category]}.csv"
    )

    # Separate positive and negative examples
    left_examples = pd.read_csv(left_examples_path)
    right_examples = pd.read_csv(right_examples_path)

    # Sample the required number of positive and negative examples
    sampled_left = left_examples.sample(num_left_category, random_state=1234)
    sampled_right = right_examples.sample(num_right_category, random_state=1234)

    # Combine and shuffle the sampled examples
    examples = (
        pd.concat([sampled_left, sampled_right])
        .sample(frac=1, random_state=1234)
        .reset_index(drop=True)
    )

    examples_str = ""
    for i in range(K):
        examples_str += prompt_method(examples.iloc[i], columns, dataset_type)
        label = examples.iloc[i]["label_str"]
        examples_str += (
            f"\nAre {dataset_type} A and {dataset_type} B the same? {label}\n\n"
        )

    return examples_str


def prompt_peeters(
    data: str,
    category: str = "Entity",
    title_col: str = "title",
    improvement: str = "basic",
    focus_attributes: list = None,
    examples: str = "",
):
    instruction = f"Do the two {category.lower()} descriptions refer to the same real-world {category.lower()}? Answer with 'Yes' if they do and 'No' if they do not.\n"

    if examples:
        final_prompt = f"{instruction} {examples} {data}"
    else:
        final_prompt = f"{instruction}{data}"

    return [{"role": "system", "content": final_prompt}]


def prompt_narayan(
    data: str,
    category: str = "Entity",
    title_col: str = "title",
    improvement: str = "basic",
    focus_attributes: list = None,
    examples: str = "",
):
    instruction = (
        f"Your task is to determine if {category} A and {category} B are the same. \n\n"
    )
    instruction = "Are Product A and Product B the same? Yes or No?"
    query = "Are Product A and Product B the same?"

    if examples:
        final_prompt = f"{instruction} {examples} {data} {query}"
    else:
        final_prompt = f"{instruction} {data} {query}"

    return [{"role": "system", "content": final_prompt}]


def prompt(
    data: str,
    category: str = "Entity",
    title_col: str = "title",
    improvement: str = "basic",
    focus_attributes: list = None,
    examples: str = "",
):
    imp = ""
    if improvement == "lenient":
        imp = "Be lenient in your judgement. "
    elif improvement == "critical":
        imp = "Be critical in your judgement. "
    elif improvement == "focus":
        imp = f"Focus more on the {title_col} attribute. "
    elif improvement == "step":
        imp = "Let's think step by step. "

    # instruction = f"Your task is to determine if {category} A and {category} B are the same. \n\n"
    instruction = f"Are {category} A and {category} B the same? Yes or No?"
    query = f"Are {category} A and {category} B the same?"
    # query = f"\n\nAre {category} A and {category} B the same? Yes or No?"

    final_prompt = instruction
    if examples:
        final_prompt += f" {examples} "

    final_prompt += f" {data} "

    if improvement != "basic":
        final_prompt += f"{imp}"

    final_prompt += query

    return [{"role": "system", "content": final_prompt}]


def extract_answer(pred):
    if pred.startswith("yes"):
        return "yes"
    else:
        return "no"
