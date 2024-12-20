import logging
import gzip
import pickle
import pandas as pd
from math import exp
import os

from utils.constants import *

logger = logging.getLogger(__name__)


# Softmax function for logprobs
def softmax(logprobs):
    exp_probs = {label: exp(lp) for label, lp in logprobs.items()}
    total = sum(exp_probs.values())
    return {label: prob / total for label, prob in exp_probs.items()}


def strip_data(data):
    # Remove backticks and single quotes from the data
    data = data.apply(
        lambda x: x.replace("`", "").replace("'", "") if isinstance(x, str) else x
    )
    # Also replace spaces at the start of the strings
    data = data.apply(lambda x: x.strip() if isinstance(x, str) else x)

    return data


def get_settings(args):
    # Get the model to use
    model = MODEL_NAMES[args.llm]
    dataset = DATASET_PATH[args.dataset]
    prompt_method = PROMPT_METHODS[args.prompt_format]
    title_col = DATASET_TITLES[args.dataset]
    dataset_type = DATASET_TYPE[args.dataset]
    focus_attributes = FOCUS_ATTRIBUTES[args.dataset]

    return model, dataset, prompt_method, title_col, dataset_type, focus_attributes


def get_data(dataset, do_train=False):
    # Load the data
    if do_train:
        return load_data(DATASET_PATH[dataset], do_train)
    else:
        CUSTOM_SELECTION = ["dblp_acm_dirty", "dblp_gs_dirty", "walmart_dirty"]
        candidate_pairs = load_data(DATASET_PATH[dataset], do_train)

        if dataset in CUSTOM_SELECTION:
            print("Dataset is in selection")
            file_path = f"custom_sized_datasets/{dataset}_complete.csv"
            candidate_pairs = pd.read_csv(file_path)
        else:
            file_path = f"input/manheim_data/{MANHEIM_DATASETS[dataset]}"

            with gzip.open(file_path, "rb") as f:
                manheim_data = pickle.load(f)

            candidate_pairs = reduce_to_manheim_subset(
                candidate_pairs, manheim_data[["id_left", "id_right"]]
            )

        return candidate_pairs


def get_file_path(args, num_runs, suffix=None):
    logging.info("Getting file path")
    # Save the results to a file
    subfolder = f"output_{args.improvement}"
    if args.context:
        subfolder += "_ctx"

    base_file_path = os.path.join(
        args.output_folder,
        subfolder,
        args.dataset,
        f"{args.llm}_k_{args.k}_n_{num_runs}_{args.prompt_format}",
    )

    if suffix is not None:
        base_file_path += f"_{suffix}"

    if args.category_pair is not None:
        cat_pair = (
            args.category_pair.replace("(", "").replace(")", "").replace(",", "_")
        )
        ratio_str = args.category_distribution.replace("/", "_")
        base_file_path = os.path.join(
            args.output_folder,
            subfolder,
            args.dataset,
            cat_pair,
            ratio_str,
            f"{args.llm}_k_{args.k}_n_{num_runs}_{args.prompt_format}",
        )

    seq = 0
    file_path = f"{base_file_path}_{'{:02d}'.format(seq)}.json"

    while os.path.exists(file_path):
        seq += 1
        file_path = f"{base_file_path}_{'{:02d}'.format(seq)}.json"

    return file_path


def get_all_cols(test_pairs) -> list:
    # Remove suffixes added to column names
    column_names = test_pairs.columns.tolist()
    cleaned_column_names = [
        name.removesuffix("_x").removesuffix("_y") for name in column_names
    ]

    # define the columns to remove
    cols_to_remove = ["id", "label", "ltable_id", "rtable_id", "label_str"]

    all_columns = [col for col in cleaned_column_names if col not in cols_to_remove]
    all_columns = list(dict.fromkeys(all_columns))

    return all_columns


def load_data(dataset: str, train=False) -> pd.DataFrame:
    logger.info(f"Loading data from {dataset}")

    tableA = pd.read_csv(f"{dataset}/TableA.csv")
    tableB = pd.read_csv(f"{dataset}/TableB.csv")

    # Use the train dataset if selected, otherwise use the test set
    test_or_train = "train" if train else "test"
    pairs = pd.read_csv(f"{dataset}/{test_or_train}.csv")

    # Add an 'order' column to preserve the original order
    pairs.insert(0, "order", range(len(pairs)))

    # Merge the tables using the 'ltable_id' and 'rtable_id' columns
    candidate_pairs = pairs.merge(tableA, left_on="ltable_id", right_on="id").merge(
        tableB, left_on="rtable_id", right_on="id"
    )

    # Sort by the 'order' column and drop it as it's no longer needed
    candidate_pairs.sort_values(by="order", inplace=True)
    candidate_pairs.drop(columns=["order"], inplace=True)

    # Save the merged and sorted data to a new CSV file
    candidate_pairs.to_csv(f"{dataset}/merged_data.csv", index=False)

    # Insert a column with "Yes/No" labels based on the "label" column
    candidate_pairs.insert(
        0,
        "label_str",
        candidate_pairs["label"].apply(lambda x: "Yes" if x == 1 else "No"),
    )

    return candidate_pairs


def reduce_to_manheim_subset(
    full_data: pd.DataFrame, manheim_subset: pd.DataFrame
) -> pd.DataFrame:
    """
    Reduce the full dataset to the intersection with the manheim_subset based on id matching.

    Args:
        full_data (pd.DataFrame): The full dataset returned by `load_data`.
        manheim_subset (pd.DataFrame): The manheim_subset dataset with 'id_left' and 'id_right' columns.

    Returns:
        pd.DataFrame: The reduced dataset, containing only rows from the full_data that match the manheim_subset.
    """
    # Make a copy of manheim_subset to avoid SettingWithCopyWarning
    manheim_subset = manheim_subset.copy()

    # Ensure the manheim_subset column names match those in full_data
    manheim_subset["ltable_id"] = manheim_subset["id_left"].apply(
        lambda x: int(x.split("_")[1])
    )
    manheim_subset["rtable_id"] = manheim_subset["id_right"].apply(
        lambda x: int(x.split("_")[1])
    )

    # Merge full_data with manheim_subset to get the intersection
    merged_data = full_data.merge(
        manheim_subset, on=["ltable_id", "rtable_id"], how="inner"
    )

    # Check for any rows in manheim_subset that do not exist in the full_data
    if len(merged_data) != len(manheim_subset):
        missing_rows = manheim_subset[
            ~manheim_subset.set_index(["ltable_id", "rtable_id"]).index.isin(
                merged_data.set_index(["ltable_id", "rtable_id"]).index
            )
        ]
        raise ValueError(
            f"Some rows in manheim_subset could not be found in the full dataset: {missing_rows}"
        )

    # Remove the id_left and id_right columns
    merged_data.drop(columns=["id_left", "id_right"], inplace=True)

    return merged_data
