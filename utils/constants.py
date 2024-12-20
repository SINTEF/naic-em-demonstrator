from utils.prompt_templates import *

FOCUS_ATTRIBUTES = {
    "beer": ["Beer_Name", "Brew_Factory_Name", "Style"],
    "itunes": ["Song_Name", "Price", "Time"],
    "itunes_dirty": ["Song_Name", "Artist_Name", "Price", "Time", "Released"],
    "fodors": ["name", "phone"],
    "dblp_acm": ["title", "authors"],
    "dblp_acm_dirty": ["title", "authors"],
    "dblp_gs": ["title", "authors"],
    "dblp_gs_dirty": ["title", "authors"],
    "amazon": ["title", "manufacturer"],
    "abt-buy": ["name", "description"],
    "walmart_dirty": ["title", "modelno"],
    "walmart": ["title", "modelno"],
}

MANHEIM_DATASETS = {
    "abt-buy": "abt-buy-sampled_gs.pkl.gz",
    "amazon": "amazon-google-sampled_gs.pkl.gz",
    "dblp_acm": "dblp-acm-sampled_gs.pkl.gz",
    "dblp_gs": "dblp-scholar-sampled_gs.pkl.gz",
    "walmart": "walmart-amazon-sampled_gs.pkl.gz",
    "wdc": "wdc-sampled_gs.pkl.gz",
}

# Logging format
LOG_FORMAT = (
    "       " + "\033[94m"
    "%(levelname)s"
    "\033[0m"
    "     "
    "\033[92m"
    "%(asctime)s"
    "\033[0m"
    " "
    "[\033[95m"
    "%(name)s"
    "\033[0m]"
    "  \t"
    "[\033[94m"
    "%(levelname)s"
    "\033[0m]"
    " - "
    "%(message)s"
)

# Path to the datasets
DATASET_PATH = {
    "beer": "input/Structured/Beer",
    "fodors": "input/Structured/Fodors-Zagats",
    "walmart": "input/Structured/Walmart-Amazon",
    "itunes": "input/Structured/iTunes-Amazon",
    "dblp_acm": "input/Structured/DBLP-ACM",
    "dblp_acm_dirty": "input/Dirty/DBLP-ACM",
    "dblp_gs": "input/Structured/DBLP-GoogleScholar",
    "dblp_gs_dirty": "input/Dirty/DBLP-GoogleScholar",
    "amazon": "input/Structured/Amazon-Google",
    "abt-buy": "input/Textual/Abt-Buy",
    "walmart": "input/Structured/Walmart-Amazon",
    "walmart_dirty": "input/Dirty/Walmart-Amazon",
    "company": "input/Textual/Company",
    "itunes_dirty": "input/Dirty/iTunes-Amazon",
}

# Dataset to type mapping (product, restaurant, etc.)
DATASET_TYPE = {
    "beer": "Product",
    "fodors": "Restaurant",
    "walmart": "Product",
    "itunes": "Song",
    "itunes_dirty": "Song",
    "dblp_acm": "Publication",
    "dblp_acm_dirty": "Publication",
    "dblp_gs": "Publication",
    "dblp_gs_dirty": "Publication",
    "amazon": "Product",
    "abt-buy": "Product",
    "walmart_dirty": "Product",
    "company": "Company",
}

COLUMNS = {
    "beer": ["Beer_Name", "Brew_Factory_Name", "Style", "ABV"],
    "itunes": [
        "Song_Name",
        "Artist_Name",
        "Album_Name",
        "Genre",
        "Price",
        "CopyRight",
        "Time",
        "Released",
    ],
    "itunes_dirty": [
        "Song_Name",
        "Artist_Name",
        "Album_Name",
        "Genre",
        "Price",
        "CopyRight",
        "Time",
        "Released",
    ],
    "fodors": ["name", "addr", "city", "phone", "type", "class"],
    "dblp_acm": ["title", "authors", "venue", "year"],
    "dblp_acm_dirty": ["title", "authors", "venue", "year"],
    "dblp_gs": ["title", "authors", "venue", "year"],
    "dblp_gs_dirty": ["title", "authors", "venue", "year"],
    "amazon": ["title", "manufacturer", "price"],
    "abt-buy": ["name", "description", "price"],
    "walmart_dirty": ["title", "category", "modelno", "price"],
    "walmart": ["title", "category", "modelno", "price"],
}

MODEL_NAMES = {
    # "gpt-3": "gpt-3.5-turbo",
    "gpt-3": "gpt-3.5-turbo-0125",
    # "gpt-3": "gpt-3.5-turbo-1106",
    "gpt-4": "gpt-4-1106-preview",
    "gpt-4o": "gpt-4o-2024-08-06",
    # "gpt-o1": "o1-preview-2024-09-12",
    # "gpt-o1-mini": "o1-mini-2024-09-12",
}

PROMPT_METHODS = {
    "natural": get_prompt_natural_language,
    "indented": get_prompt_indented,
    "semicolon": get_prompt_semicolon_delimiter,
    "tabular": get_prompt_tabular_format,
    "basic": get_prompt_basic,
    "delim": get_prompt_indented_delimiters,
    "json": get_prompt_json_format,
    "xml": get_prompt_xml_format,
    "html": get_prompt_html_format,
    "yaml": get_prompt_yaml_format,
    "angle": get_prompt_angle_brackets,
    "natural_messy": get_prompt_natural_language_messy,
    "indented_messy": get_prompt_indented_messy,
    "tabular_messy": get_prompt_tabular_format_messy,
    "step": get_prompt_step_format,
    "gpt": get_prompt_gpt_format,
    "step_natural": get_prompt_step_natural_format,
    "step_short": get_prompt_step_short_format,
    "no_attr": get_prompt_no_attributes,
    "narayan": get_prompt_narayan,
    "peeters": get_prompt_peeters,
}

DATASET_NAMES = {
    "beer": "BeerAdvo-RateBeer",
    "fodors": "Fodors-Zagats",
    "walmart": "Walmart-Amazon",
    "itunes": "iTunes-Amazon",
    "dblp": "DBLP-ACM",
    "dblp_acm": "DBLP-ACM",
    "dblp_acm_dirty": "DBLP-ACM [Dirty]",
    "dblp_gs": "DBLP-GoogleScholar",
    "dblp_gs_dirty": "DBLP-GoogleScholar [Dirty]",
    "amazon": "Amazon-Google",
    "abt-buy": "Abt-Buy",
    "walmart_dirty": "Walmart-Amazon [Dirty]",
    "walmart": "Walmart-Amazon",
    "company": "Company",
    "itunes_dirty": "iTunes-Amazon [Dirty]",
}

DATASET_TITLES = {
    "beer": "Beer_Name",
    "fodors": "name",
    "walmart": "title",
    "walmart_dirty": "title",
    "itunes": "Song_Name",
    "abt-buy": "name",
    "dblp_acm": "title",
    "dblp_acm_dirty": "title",
    "dblp_gs": "title",
    "dblp_gs_dirty": "title",
    "amazon": "title",
    "abt-buy": "title",
    "itunes_dirty": "Song_Name",
}

PROMPT_FORMATS = [
    "json",
    "natural",
    "tabular",
    "yaml",
    "no_attr",
    "narayan",
    "peeters",
    "basic",
    "step",
    "step_short",
    "step_natural",
]

CATEGORY_DICT = {
    "basic": "Basic",
    "basic_ctx": "Basic + Context",
    "critical": "Critical",
    "critical_ctx": "Critical + Context",
    "lenient": "Lenient",
    "focus": "Focus Attribute",
    "lenient_ctx": "Lenient + Context",
    "baseline": "Baseline",
}

CATEGORIES = [
    "basic",
    "lenient",
    "critical",
    "focus",
    "step",
]

DATASET_SIZES = {
    # "beer": 500,
    # "fodors": 500,
    "walmart": 2049,
    # "itunes": 500,
    "dblp": 2473,
    "dblp_acm_dirty": 2473,
    # "dblp_gs": 5742,
    "dblp_gs": 2500,
    "dblp_gs_dirty": 5742,
    "amazon": 2293,
    "abt-buy": 1916,
    "walmart_dirty": 2049,
    # "company": 500,
    # "itunes_dirty": 500,
}

DATASET_TRAIN_SIZES = {
    # "beer": 500,
    # "fodors": 500,
    "walmart": 2049,
    # "itunes": 500,
    "dblp": 2473,
    "dblp_acm_dirty": 2473,
    # "dblp_gs": 5742,
    "dblp_gs": 2500,
    "dblp_gs_dirty": 5742,
    "amazon": 6874,
    "abt-buy": 1916,
    "walmart_dirty": 2049,
    # "company": 500,
    # "itunes_dirty": 500,
}

NEGATIVE_SAMPLE_FILES = {
    "abt-buy": "category_selections/abt-buy/abt-buy_neg_pairs.csv",
    "amazon": "category_selections/amazon/amazon_neg_pairs.csv",
    "dblp_acm": "category_selections/dblp_acm/dblp_acm_neg_pairs.csv",
    "dblp_gs": "category_selections/dblp_gs/dblp_gs_neg_pairs.csv",
    "dblp_acm_dirty": "category_selections/dblp_acm_dirty/dblp_acm_dirty_neg_pairs.csv",
    "dblp_gs_dirty": "category_selections/dblp_gs_dirty/dblp_gs_dirty_neg_pairs.csv",
    "walmart": "category_selections/walmart/walmart_neg_pairs.csv",
}

POSITIVE_SAMPLE_FILES = {
    "abt-buy": "category_selections/abt-buy/abt-buy_pos_pairs.csv",
    "amazon": "category_selections/amazon/amazon_pos_pairs.csv",
    "dblp_acm": "category_selections/dblp_acm/dblp_acm_pos_pairs.csv",
    "dblp_gs": "category_selections/dblp_gs/dblp_gs_pos_pairs.csv",
    "dblp_acm_dirty": "category_selections/dblp_acm_dirty/dblp_acm_dirty_pos_pairs.csv",
    "dblp_gs_dirty": "category_selections/dblp_gs_dirty/dblp_gs_dirty_pos_pairs.csv",
    "walmart": "category_selections/walmart/walmart_pos_pairs.csv",
}

CATEGORY_PAIRS = {
    "TP": "True Positive",
    "TN": "True Negative",
    "FP": "False Positive",
    "FN": "False Negative",
    "I": "Incorrect",
    "C": "Correct",
    "P": "Positive",
    "N": "Negative",
}

CATEGORY_ID = {
    "TP": "tp",
    "TN": "tn",
    "FP": "fp",
    "FN": "fn",
    "I": "inc",
    "C": "crc",
    "P": "pos",
    "N": "neg",
}


MANHEIM_DATASET_SIZES = {
    "abt-buy": 1206,
    "amazon": 1234,
    "dblp_acm": 1250,
    "dblp_gs": 1250,
    "dblp_gs_dirty": 1250,
    "dblp_acm_dirty": 1250,
    "walmart_dirty": 1250,
    "walmart": 1193,
    # "wdc": ,
}
