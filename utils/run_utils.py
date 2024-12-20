import logging
from typing import Tuple
import openai

from utils.data_utils import *
from utils.constants import *
from utils.prompt_utils import *


logger = logging.getLogger(__name__)


def compare_pairs(args, candidate_pairs, examples, prompt_method, dataset_type):
    # List for storing the predictions
    preds = []

    # Dictionary for storing the metrics
    trial_metrics = {
        "prec": [],
        "rec": [],
        "f1": [],
        "acc": [],
        "time": [],
        "details": [],
    }

    model = MODEL_NAMES[args.llm]
    title_col = DATASET_TITLES[args.dataset]
    focus_attributes = FOCUS_ATTRIBUTES[args.dataset]
    # columns = COLUMNS[args.dataset]

    total_pairs = min(args.num_pairs, len(candidate_pairs))
    # Go through each pair of candidates and prompt the model for a prediction
    for i in range(total_pairs):
        logging.info(f"Run {i+1}/{total_pairs}")

        data = candidate_pairs.iloc[i]

        # Strip the data if the prompt format is not 'narayan'
        if args.prompt_format != "narayan":
            data = strip_data(data)

        # Get the prompt in the desired format
        prompt_data = prompt_method(data, COLUMNS[args.dataset], dataset_type)

        if args.prompt_format == "narayan":
            messages = prompt_narayan(
                prompt_data, dataset_type, title_col, args.improvement, focus_attributes
            )
        elif args.prompt_format == "peeters":
            messages = prompt_peeters(
                prompt_data, dataset_type, title_col, args.improvement, focus_attributes
            )

        else:
            messages = prompt(
                prompt_data,
                dataset_type,
                title_col,
                args.improvement,
                focus_attributes,
                examples,
            )
        prompt_str = messages[0]["content"]

        # Log the prompt
        log_multiline_message(logger, prompt_str, level=logging.INFO, indent=14)

        try:
            # Get the completion from the Model
            response_raw, logprob = get_completion_from_messages(messages, model)
            # Make sure the answer is Yes or No
            pred = extract_answer(response_raw.lower())
            if not args.chain_of_thought:
                preds.append(pred)

            if args.chain_of_thought:
                cot_message = [
                    {
                        "role": "system",
                        "content": f"Can you extract the Yes or No answer from the following text? '{response_raw}'",
                    }
                ]
                cot_response_raw, cot_logprob = get_completion_from_messages(
                    cot_message, model
                )
                cot_pred = extract_answer(cot_response_raw.lower())
                preds.append(cot_pred)

                # Store the response in the trial metrics

            # Set the predicted label in the candidate_pairs DataFrame
            candidate_pairs.at[i, "predicted_label"] = pred
            candidate_pairs.iloc[
                i, candidate_pairs.columns.get_loc("predicted_label")
            ] = pred

            # The ground truth label in string format
            actual_label = candidate_pairs.iloc[i]["label_str"].lower()

            # Log the prediction vs. actual
            logger.info(f"Prediction: {pred} | Actual: {actual_label}")

            confidence = 0.0

            if logprob:
                softmax_probs = softmax(logprob)
                first_label = next(iter(logprob))
                confidence = softmax_probs.get(first_label, 0.0)

            if args.chain_of_thought:
                trial_metrics["details"].append(
                    {
                        "prompt": prompt_str,
                        # "predicted_label": pred,
                        "actual_label": actual_label,
                        "response_raw": response_raw,
                        "initial_prediction": pred,
                        "logprob": logprob,
                        "confidence": confidence,
                        "cot_prompt": cot_message[0]["content"],
                        "cot_response_raw": cot_response_raw,
                        "predicted_label": cot_pred,
                        "cot_logprob": cot_logprob,
                    }
                )
            else:
                trial_metrics["details"].append(
                    {
                        "prompt": prompt_str,
                        "predicted_label": pred,
                        "actual_label": actual_label,
                        "response_raw": response_raw,
                        "logprob": logprob,
                        "confidence": confidence,
                    }
                )
            # Update the progress in the database
            file_path = "test"

        except openai.OpenAIError as e:
            logging.error(f"Service Unavailable Error: {e}")
            logging.info("Hello we are here")
            continue

    return trial_metrics, preds


def compute_metrics(preds: List, golds: List, test_pairs):
    """Compute metrics."""
    mets = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "crc": 0, "total": 0}
    # preds = test_pairs["label_str"]
    for pred, label in zip(preds, golds):
        label = label.strip().lower()
        pred = pred.strip().lower()
        # print(f"Pred: {pred}")
        # print(f"Label: {label}")
        mets["total"] += 1
        crc = pred.startswith(label)
        # Measure equal accuracy for generation
        if crc:
            mets["crc"] += 1
        if label == "yes":
            if crc:
                mets["tp"] += 1
            else:
                mets["fn"] += 1
        elif label == "no":
            if crc:
                mets["tn"] += 1
            else:
                mets["fp"] += 1

    prec = mets["tp"] / max(1, (mets["tp"] + mets["fp"]))
    rec = mets["tp"] / max(1, (mets["tp"] + mets["fn"]))
    acc = mets["crc"] / mets["total"]
    f1 = 2 * prec * rec / max(1, (prec + rec))
    return prec, rec, acc, f1


def get_trial_metrics(candidate_pairs, num_pairs, trial_metrics, preds, execution_time):
    gt = candidate_pairs["label_str"][:num_pairs]
    prec, rec, acc, f1 = compute_metrics(preds, gt, candidate_pairs)

    trial_metrics["rec"].append(rec)
    trial_metrics["prec"].append(prec)
    trial_metrics["acc"].append(acc)
    trial_metrics["f1"].append(f1)
    trial_metrics["time"].append(execution_time)

    return trial_metrics


def log_metrics(trial_metrics):
    # Create a dictionary for logging that doesn't include the 'details'
    log_metrics = {
        key: value for key, value in trial_metrics.items() if key != "details"
    }

    # Log the results excluding 'details'
    logger.info(f"Final Metrics {json.dumps(log_metrics, indent=16)}")


def write_trial_metrics_to_file(trial_metrics, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as outfile:
        # Use the dump() method to write the dictionary to the file in JSON format
        json.dump(trial_metrics, outfile, indent=5)

    return file_path


def log_multiline_message(logger, message, level=logging.INFO, indent=4):
    """
    Logs a multiline message with specified indentation for all lines.

    Args:
    - logger: The logging.Logger instance to log the message.
    - message: The multiline string message to log.
    - level: The logging level (e.g., logging.INFO, logging.ERROR).
    - indent: The number of spaces to indent each line.
    """
    # Split the message into lines
    lines = message.splitlines()
    start = "\n<<<<<<<<<<< PROMPT START >>>>>>>>>>>\n"
    end = "\n<<<<<<<<<<< PROMPT END >>>>>>>>>>>\n\n"
    lines = [start] + lines + [end]

    # Indent each line
    indented_lines = [(" " * indent) + line for line in lines]

    # Join the indented lines back into a single string
    indented_message = "\n".join(indented_lines)

    # Log the indented message
    if level == logging.DEBUG:
        logger.debug(indented_message)
    elif level == logging.INFO:
        logger.info(indented_message)
    elif level == logging.WARNING:
        logger.warning(indented_message)
    elif level == logging.ERROR:
        logger.error(indented_message)
    elif level == logging.CRITICAL:
        logger.critical(indented_message)
    else:
        logger.log(level, indented_message)


def calculate_tp_fp_tn_fn(results, threshold):
    """
    Calculate True Positives (TP), False Positives (FP), True Negatives (TN),
    and False Negatives (FN) for a given confidence threshold.

    Args:
    - results: List of dictionaries containing actual and predicted labels, and confidence scores.
    - threshold: The confidence threshold to apply for predictions.

    Returns:
    - tp: True Positives count.
    - fp: False Positives count.
    - tn: True Negatives count.
    - fn: False Negatives count.
    """
    tp = fp = tn = fn = 0

    # Iterate through the results and apply the confidence threshold
    for result in results:
        actual_label = result["actual_label"]
        predicted_label = result["predicted_label"]
        confidence = result["confidence"]

        # Apply threshold: if confidence >= threshold, we trust the predicted label
        if confidence >= threshold:
            final_pred = predicted_label
        else:
            # If confidence is below the threshold, we consider the opposite prediction
            final_pred = "no" if predicted_label == "yes" else "yes"

        # Count TP, FP, TN, FN based on the actual and final predicted labels
        if actual_label == "yes" and final_pred == "yes":
            tp += 1
        elif actual_label == "no" and final_pred == "yes":
            fp += 1
        elif actual_label == "no" and final_pred == "no":
            tn += 1
        elif actual_label == "yes" and final_pred == "no":
            fn += 1

    return tp, fp, tn, fn


def category_tuple_from_string(category_pair) -> Tuple[str, str]:
    """
    Convert a string of the form "(A,B)" to a tuple of strings ("A", "B").
    """
    return tuple(map(str.strip, category_pair[1:-1].split(",")))
