import time
import logging
import argparse

from utils.constants import *
from utils.data_utils import *
from utils.prompt_utils import *
from utils.run_utils import *

# Set up the logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


# Parsing arguments
def read_args():
    parser = argparse.ArgumentParser(
        description="Run entity matching using GPT models on the Magellan dataset"
    )

    parser.add_argument(
        "-o", "--output_folder", help="The folder to put results in", default="output"
    )

    parser.add_argument(
        "-d",
        "--dataset",
        choices=list(DATASET_NAMES.keys()),
        help="The dataset to test",
        default="amazon",
    )

    parser.add_argument(
        "-ctx",
        "--context",
        action="store_true",
        help="Add subtle context to prompt",
        default=False,
    )

    parser.add_argument(
        "-cot",
        "--chain_of_thought",
        action="store_true",
        help="Ask the model to extract 'yes' or 'no' from the prompt",
        default=False,
    )

    parser.add_argument(
        "-ctp",
        "--category_pair",
        help="The pair of categories to use for the examples. Example: (TN,FP)",
        default=None,
    )

    parser.add_argument(
        "-cd",
        "--category_distribution",
        help="Decides distribution of examples from the two categories (specified by the category_pair argument. Ex: '30/70'",
        default="50/50",
    )

    parser.add_argument(
        "-k",
        "--k",
        type=int,
        help="Number of examples to include in the prompt",
        default=0,
    )

    parser.add_argument(
        "-n",
        "--num_pairs",
        type=int,
        help="Number of pairs to test",
        default=1,
    )

    parser.add_argument(
        "-llm",
        "--llm",
        choices=list(MODEL_NAMES.keys()),
        help="Which model to use",
        default="gpt-4",
    )

    parser.add_argument(
        "-imp",
        "--improvement",
        choices=CATEGORIES,
        help="Improve the model",
        default="basic",
    )

    parser.add_argument(
        "-dt",
        "--do_train",
        action="store_true",
        help="Use the training dataset for matching",
        default=False,
    )

    parser.add_argument(
        "-pf",
        "--prompt_format",
        choices=PROMPT_FORMATS,
        help="The prompt format to use",
        default="natural",
    )

    args = parser.parse_args()

    return args


DATASET_SELECTION = ["dblp_acm_dirty", "dblp_gs_dirty", "walmart_dirty"]


def run_entity_matching(args):
    # Get the settings
    columns = COLUMNS[args.dataset]
    dataset = DATASET_PATH[args.dataset]
    if args.k > 0 and args.category_pair is not None:
        category_tuple = category_tuple_from_string(args.category_pair)
    else:
        category_tuple = None
    prompt_method = PROMPT_METHODS[args.prompt_format]

    # Use 'Entity' as default, otherwise use the dataset type (e.g., 'Product' or 'Publications' etc.)
    dataset_type = DATASET_TYPE[args.dataset] if args.context else "Entity"

    candidate_pairs = get_data(args.dataset, args.do_train)

    # Number of pairs can't be more than the size of the dataset
    num_pairs = min(args.num_pairs, len(candidate_pairs))

    # Start the timer
    start_time = time.perf_counter()

    # Get the examples for the prompt
    examples = ""
    if args.k > 0:
        examples = get_examples(
            prompt_method=prompt_method,
            columns=columns,
            dataset_type=dataset_type,
            K=args.k,
            dataset=args.dataset,
            category_pair=category_tuple,
            ratio=args.category_distribution,
        )

        logging.info(f"Examples: {examples}")

    logging.info(f"Running the matching on {num_pairs} examples")

    trial_metrics, preds = compare_pairs(
        args, candidate_pairs, examples, prompt_method, dataset_type
    )

    # End the timer
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    trial_metrics = get_trial_metrics(
        candidate_pairs, num_pairs, trial_metrics, preds, execution_time
    )

    # log metrics
    log_metrics(trial_metrics)

    # Write the trial metrics to a JSON file
    file_path = get_file_path(args, num_pairs)
    write_trial_metrics_to_file(trial_metrics, file_path)

    return file_path


# Entity Matching Script
def main():
    args = read_args()
    print(args)
    file_path = run_entity_matching(args)

    logger.info(f"Saved results to {file_path}")


if __name__ == "__main__":
    main()
