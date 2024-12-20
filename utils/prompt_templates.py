from typing import List
import pandas as pd
import json
import random


# Normal natural prompt
def get_prompt_natural_language(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    """
    Construct a query from the candidate pair data for entity matching, formatting the details in a natural language description.

    Args:
    - candidate_pair (pd.Series): A single row from the candidate pairs DataFrame, representing a pair of entities to compare.
    - columns (list of str): List of column names to include in the prompt, formatted in a natural language style.

    Returns:
    - str: A query string formatted for prompting GPT, describing both entities in natural language.
    """
    # Start the description for Product A and B
    # NOTE: Change it to A is a {category} and B is a {category}?
    product_a_details = f"{category} A is"
    product_b_details = f"{category} B is"

    # Define a template for formatting each attribute in natural language
    attribute_templates = {
        "title": "titled '{}'",
        "name": "named '{}'",
        "category": "in the category '{}'",
        "brand": "from the brand '{}'",
        "modelno": "with model number '{}'",
        "price": "priced at ${}",
        "manufacturer": "manufactured by '{}'",
        "addr": "located at '{}'",
        "city": "in '{}'",
        "phone": "contactable via '{}'",
        "type": "categorized as '{}'",
        "class": "with a class rating of '{}'",
        "description": "described as '{}'",
        "content": "with content '{}'",
    }

    # Helper function to format attribute text
    def format_attribute(col, value):
        template = attribute_templates.get(col.lower(), "{}")
        return template.format(value)

    # Iterate through each specified column
    for col in columns:
        col_x = col + "_x"
        col_y = col + "_y"

        # Format and append details for Product A
        value_a = candidate_pair.get(col_x, "")
        if pd.notna(value_a) and value_a != "":
            product_a_details += f" {format_attribute(col, value_a)},"

        # Format and append details for Product B
        value_b = candidate_pair.get(col_y, "")
        if pd.notna(value_b) and value_b != "":
            product_b_details += f" {format_attribute(col, value_b)},"

    # Remove the last comma and add the product type if known, adjust as needed
    product_a_details = product_a_details.rstrip(",") + "."
    product_b_details = product_b_details.rstrip(",") + "."

    # Construct the final query
    query = f"{product_a_details} {product_b_details}"

    return query


def get_prompt_no_attributes(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    """
    Construct a query from the candidate pair data for entity matching, without including the attribute names, only the values.

    Args:
    - candidate_pair (pd.Series): A single row from the candidate pairs DataFrame, representing a pair of entities to compare.
    - columns (list of str): List of column names to include in the prompt, without the attribute names.

    Returns:
    - str: A query string formatted for prompting GPT, describing both entities with only the attribute values.
    """
    # Start the description for Product A and B
    product_a_details = f"{category} A: "
    product_b_details = f"{category} B: "

    # Iterate through each specified column
    for col in columns:
        col_x = col + "_x"
        col_y = col + "_y"

        # Append details for Product A
        value_a = candidate_pair.get(col_x, "")
        if pd.notna(value_a) and value_a != "":
            product_a_details += f"{value_a}, "

        # Append details for Product B
        value_b = candidate_pair.get(col_y, "")
        if pd.notna(value_b) and value_b != "":
            product_b_details += f"{value_b}, "

    # Remove the last comma and add the product type if known, adjust as need
    product_a_details = product_a_details.rstrip(",") + "."
    product_b_details = product_b_details.rstrip(",") + "."

    # Construct the final Query
    query = f"{product_a_details} {product_b_details}"

    return query


# Natural prompt, same as above, but with completely random and inconsistent delimiters
def get_prompt_natural_language_messy(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    """
    Construct a query from the candidate pair data for entity matching, formatting the details in a natural language description.

    Args:
    - candidate_pair (pd.Series): A single row from the candidate pairs DataFrame, representing a pair of entities to compare.
    - columns (list of str): List of column names to include in the prompt, formatted in a natural language style.

    Returns:
    - str: A query string formatted for prompting GPT, describing both entities in natural language.
    """
    # Start the description for Product A and B
    # NOTE: Change it to A is a {category} and B is a {category}?
    product_a_details = f"{category} A is "
    product_b_details = f"{category} B is "

    # Define a template for formatting with totally random and very bad use of delimiters, (example: "titled `` {} > ", or "named ; {} ''")
    attribute_templates = {
        "title": "titled `` {} > ",
        "name": "named ; {} ''",
        "category": "in the category ## {}  #",
        "brand": "from the brand - {} ---",
        "modelno": "with model number '' {} ",
        "price": "priced at ${}",
        "manufacturer": "manufactured by ;; {} `'",
        "addr": "located at >< {} >",
        "city": "in ' {} '",
        "phone": "contactable via -' {} ```",
        "type": "categorized as   {} #",
        "class": "with a class rating of ## {} ",
        "description": "described as  {} ",
    }

    # Helper function to format attribute text
    def format_attribute(col, value):
        template = attribute_templates.get(col.lower(), "{}")
        return template.format(value)

    # Iterate through each specified column
    for col in columns:
        col_x = col + "_x"
        col_y = col + "_y"

        # Format and append details for Product A
        value_a = candidate_pair.get(col_x, "")
        if pd.notna(value_a) and value_a != "":
            product_a_details += f" {format_attribute(col, value_a)};"

        # Format and append details for Product B
        value_b = candidate_pair.get(col_y, "")
        if pd.notna(value_b) and value_b != "":
            product_b_details += f" {format_attribute(col, value_b)};"

    # Remove the last semicolon and add the product type if known, adjust as needed
    product_a_details = product_a_details.rstrip(";") + "."
    product_b_details = product_b_details.rstrip(";") + "."
    # Construct the final query
    query = f"{product_a_details} {product_b_details}"

    return query


# Natural prompt with completely random and inconsistent delimiters for each entity
def get_prompt_natural_language_messy(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    """
    Construct a query from the candidate pair data for entity matching, formatting the details in a natural language description.
    Delimiters for entity A and B are chosen randomly and can be different.

    Args:
    - candidate_pair (pd.Series): A single row from the candidate pairs DataFrame, representing a pair of entities to compare.
    - columns (list of str): List of column names to include in the prompt, formatted in a natural language style.

    Returns:
    - str: A query string formatted for prompting GPT, describing both entities in natural language with random delimiters.
    """
    random.seed(1234)
    # Start the description for Product A and B
    product_a_details = f"{category} A is "
    product_b_details = f"{category} B is "

    # Define potential delimiters
    delimiters = [
        "`` {} >",
        "; {} ''",
        "## {} #",
        "- {} ---",
        "'' {} ",
        "${}",
        ";; {} `'",
        ">< {} >",
        "' {} '",
        "-' {} ```",
        "{} #",
        "## {} ",
    ]

    # Helper function to format attribute text with random delimiters for A and B
    def format_attribute(col, value, entity):
        # Randomly select a delimiter
        delimiter = random.choice(delimiters)
        # Return formatted string using the selected delimiter
        return delimiter.format(value)

    # Iterate through each specified column
    for col in columns:
        col_x = col + "_x"
        col_y = col + "_y"

        # Format and append details for Product A with a random delimiter
        value_a = candidate_pair.get(col_x, "")
        if pd.notna(value_a) and value_a != "":
            product_a_details += f" {format_attribute(col, value_a, 'A')};"

        # Format and append details for Product B with a (potentially different) random delimiter
        value_b = candidate_pair.get(col_y, "")
        if pd.notna(value_b) and value_b != "":
            product_b_details += f" {format_attribute(col, value_b, 'B')};"

    # Construct the final query with randomized details for both entities
    query = f"{product_a_details.rstrip(';')}. {product_b_details.rstrip(';')}."

    # Add the final question
    query += (
        f" Can you determine if {category} A and {category} B are the same? Yes or No?"
    )

    return query


def get_prompt_basic(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    # Initialize details for Product A and B with the product name prefix
    product_a_details = f"{category} A is "
    product_b_details = f"{category} B is "
    # Iterate through each specified column
    for col in columns:
        col_x = col + "_x"
        col_y = col + "_y"

        # Append details for Product A, using semicolons as delimiters
        value_a = candidate_pair.get(col_x, "")
        if pd.notna(value_a) and value_a != "":
            product_a_details += f" {col.capitalize()}: {value_a},"
        else:
            product_a_details += f" {col.capitalize()}: ,"

        # Append details for Product B, using semicolons as delimiters
        value_b = candidate_pair.get(col_y, "")
        if pd.notna(value_b) and value_b != "":
            product_b_details += f" {col.capitalize()}: {value_b},"
        else:
            product_b_details += f" {col.capitalize()}: ,"

    # Remove the last semicolon for aesthetics
    # and add the closing triple backticks
    product_a_details = product_a_details.rstrip(",") + ""
    product_b_details = product_b_details.rstrip(",") + ""

    # Construct the final query
    query = f"{product_a_details}\n{product_b_details}"

    return query


def get_prompt_narayan(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    # Initialize details for Product A and B with the product name prefix
    # Example: Are Product A and Product B the same? Yes or No?  Product A is name: citrus. addr: ' 6703 melrose ave. '. city: ` los angeles '. phone: 213/857 -0034. type: californian. class: 6. Product B is name: ` le chardonnay ( los angeles ) '. addr: ' 8284 melrose ave. '. city: ` los angeles '. phone: 213-655-8880. type: ` french bistro '. class: 12. Are Product A and Product B the same?
    category = "Product"
    product_a_details = f"{category} A is"
    product_b_details = f"{category} B is"
    # Iterate through each specified column
    for col in columns:
        col_x = col + "_x"
        col_y = col + "_y"

        # Append details for Product A, using semicolons as delimiters
        value_a = candidate_pair.get(col_x, "")
        if pd.notna(value_a) and value_a != "":
            product_a_details += f" {col}: {value_a}."
        else:
            product_a_details += f" {col}: ."

        # Append details for Product B, using semicolons as delimiters
        value_b = candidate_pair.get(col_y, "")
        if pd.notna(value_b) and value_b != "":
            product_b_details += f" {col}: {value_b},"
        else:
            product_b_details += f" {col}: ."

    # Construct the final query
    query = f"{product_a_details} {product_b_details}"
    return query


def get_prompt_peeters(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    # Initialize details for Product A and B with the product name prefix
    # Example: Are Product A and Product B the same? Yes or No?  Product A is name: citrus. addr: ' 6703 melrose ave. '. city: ` los angeles '. phone: 213/857 -0034. type: californian. class: 6. Product B is name: ` le chardonnay ( los angeles ) '. addr: ' 8284 melrose ave. '. city: ` los angeles '. phone: 213-655-8880. type: ` french bistro '. class: 12. Are Product A and Product B the same?
    product_a_details = f"{category} 1: '"
    product_b_details = f"{category} 2: '"
    # Iterate through each specified column
    for col in columns:
        col_x = col + "_x"
        col_y = col + "_y"

        value_a = candidate_pair.get(col_x, "")
        if pd.notna(value_a) and value_a != "":
            product_a_details += f"{value_a} "
        else:
            product_a_details += f" ."

        value_b = candidate_pair.get(col_y, "")
        if pd.notna(value_b) and value_b != "":
            product_b_details += f"{value_b} "
        else:
            product_b_details += f" ."

    # replace final space with '
    product_a_details = product_a_details.rstrip() + "'"
    product_b_details = product_b_details.rstrip() + "'"

    # Remove
    # Construct the final query
    query = f"{product_a_details}\n{product_b_details}"

    return query


def get_prompt_semicolon_delimiter(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    """
    Construct a query from the candidate pair data for entity matching, formatting the details with semicolons, and
    triple backticks for separating each product.

    Args:
    - candidate_pair (pd.Series): A single row from the candidate pairs DataFrame, representing a pair of entities to compare.
    - columns (list of str): List of column names to include in the prompt, without '_x' or '_y' suffixes.

    Returns:
    - str: A query string formatted for prompting GPT, including fields for both entities based on the specified columns,
           using semicolons as delimiters.
    """
    # Initialize details for Product A and B with the product name prefix
    product_a_details = f"{category} A: ```"
    product_b_details = f"{category} B: ```"

    # Iterate through each specified column
    for col in columns:
        col_x = col + "_x"
        col_y = col + "_y"

        # Append details for Product A, using semicolons as delimiters
        value_a = candidate_pair.get(col_x, "")
        if pd.notna(value_a) and value_a != "":
            product_a_details += f" {col.capitalize()}: {value_a};"
        else:
            product_a_details += f" {col.capitalize()}: ;"

        # Append details for Product B, using semicolons as delimiters
        value_b = candidate_pair.get(col_y, "")
        if pd.notna(value_b) and value_b != "":
            product_b_details += f" {col.capitalize()}: {value_b};"
        else:
            product_b_details += f" {col.capitalize()}: ;"

    # Remove the last semicolon for aesthetics
    # and add the closing triple backticks
    product_a_details = product_a_details.rstrip(";") + " ```"
    product_b_details = product_b_details.rstrip(";") + " ```"

    # Construct the final query
    query = f"{product_a_details}\n{product_b_details}"

    return query


def get_prompt_indented(candidate_pair, columns, category="Product"):
    # Initialize strings for Product A and B details
    product_a_details = f"{category} A"
    product_b_details = f"{category} B"

    # Iterate through each specified column
    for col in columns:
        col_x = col + "_x"
        col_y = col + "_y"

        # Check and append details for Product A
        if col_x in candidate_pair:
            product_a_details += f"\n    {col}: {candidate_pair[col_x]}"

        # Check and append details for Product B
        if col_y in candidate_pair:
            product_b_details += f"\n    {col}: {candidate_pair[col_y]}"

    # Construct the final query
    query = f"{product_a_details}\n\n{product_b_details}"

    return query


# indented messy prompt with random delimiters
def get_prompt_indented_messy(candidate_pair, columns, category="Product"):
    # Initialize strings for Product A and B details
    product_a_details = f"{category} A"
    product_b_details = f"{category} B"

    delimiters = [
        ">",
        "<",
        ";;",
        ":",
        "!",
        "#",
        "--",
        "$",
        "%",
        "^",
        "&",
        "*",
        "(",
        ")",
        "_",
        "+",
        "=",
        "{",
        "}",
        "[",
        "]",
        "|",
        "\\",
        ":",
        ";",
        "'",
        '"',
        ",",
        ".",
        "<",
        ">",
        "?",
        "/",
    ]

    # Iterate through each specified column and add random delimiters
    i = 0
    for col in columns:
        col_x = col + "_x"
        col_y = col + "_y"

        # Check and append details for Product A
        if col_x in candidate_pair:
            product_a_details += (
                f"\n    {col}: {delimiters[i]}{candidate_pair[col_x]}{delimiters[i+1]}"
            )

        i += 1
        # Check and append details for Product B
        if col_y in candidate_pair:
            product_b_details += (
                f"\n    {col}: {delimiters[i]}{candidate_pair[col_y]}{delimiters[i+1]}"
            )

        i += 1

    # Construct the final query
    query = f"{product_a_details}\n\n{product_b_details}"

    return query


def get_prompt_tabular_format(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    """
    Construct a query from the candidate pair data for entity matching, formatting the details in a tabular layout.
    The column headers and rows are generated dynamically based on the provided columns list.

    Args:
    - candidate_pair (pd.Series): A single row from the candidate pairs DataFrame, representing a pair of entities to compare.
    - columns (list of str): List of column names to include in the table, without '_x' and '_y' suffixes.

    Returns:
    - str: A query string formatted for prompting GPT, describing both entities in a tabular layout with dynamic headers.
    """

    # Create the header dynamically
    headers = [col.capitalize() for col in columns]
    headers.insert(0, f"{category}")
    header_line = " | ".join(headers)
    separator = "-" * (
        len(header_line) + 3 * (len(headers) - 1)
    )  # Adjust based on the number of columns

    # Initialize rows for Product A and B
    product_a_values = ["A"]
    product_b_values = ["B"]

    # Collect values for each column
    for col in columns:
        value_a = str(candidate_pair.get(f"{col}_x", "N/A"))
        value_b = str(candidate_pair.get(f"{col}_y", "N/A"))
        product_a_values.append(value_a)
        product_b_values.append(value_b)

    # Format the rows for Product A and B
    product_a_row = " | ".join(product_a_values)
    product_b_row = " | ".join(product_b_values)

    # Combine header, separator, and rows into the final query
    query = (
        f"{category} Comparison:\n{header_line}\n{separator}\n"
        f"{product_a_row}\n{product_b_row}\n"
    )

    return query


# Function to join values with varying separators
def join_with_various_separators(values, separators):
    # Ensure there are enough separators for the values

    # Initialize the result string with the first value
    result = values[0]

    # Iterate over the values and separators to construct the result string
    for value, separator in zip(values[1:], separators):
        result += separator + value

    return result


# Tabular format messy with random delimiters
def get_prompt_tabular_format_messy(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    """
    Construct a query from the candidate pair data for entity matching, formatting the details in a tabular layout.
    The column headers and rows are generated dynamically based on the provided columns list.

    Args:
    - candidate_pair (pd.Series): A single row from the candidate pairs DataFrame, representing a pair of entities to compare.
    - columns (list of str): List of column names to include in the table, without '_x' and '_y' suffixes.

    Returns:
    - str: A query string formatted for prompting GPT, describing both entities in a tabular layout with dynamic headers.
    """

    # Create the header dynamically
    headers = [col.capitalize() for col in columns]
    headers.insert(0, f"{category}")
    header_line = " | ".join(headers)
    separator = "-" * (
        len(header_line) + 3 * (len(headers) - 1)
    )  # Adjust based on the number of columns

    delimiters = [
        ">",
        "<",
        ";;",
        ":",
        "!",
        "#",
        "--",
        "$",
        "%",
        "^",
        "&",
        "*",
        "(",
        ")",
        "_",
        "+",
        "=",
        "{",
        "}",
        "[",
        "]",
        "|",
        "\\",
        ":",
        ";",
        "'",
        '"',
        ",",
        ".",
        "<",
        ">",
        "?",
        "/",
    ]

    # Initialize rows for Product A and B
    product_a_values = ["A"]
    product_b_values = ["B"]

    # Collect values for each column
    # i = 0
    # for col in columns:
    #     value_a = str(candidate_pair.get(f"{col}_x", "N/A"))
    #     value_b = str(candidate_pair.get(f"{col}_y", "N/A"))
    #     product_a_values.append(f"{delimiters[i]}{value_a}{delimiters[i+1]}")
    #     product_b_values.append(f"{delimiters[i]}{value_b}{delimiters[i+1]}")
    #     i += 2

    # Collect values for each column
    for col in columns:
        value_a = str(candidate_pair.get(f"{col}_x", "N/A"))
        value_b = str(candidate_pair.get(f"{col}_y", "N/A"))
        product_a_values.append(value_a)
        product_b_values.append(value_b)

    separators = [
        " | ",
        " | | ",
        " ",
        " | ",
        " ",
        " ",
        " | ",
        " | | ",
        " | ",
        " || ",
        " ",
        " | ",
        " ",
        " ",
        " | ",
        " | | ",
    ]

    # Format the rows for Product A and B with varying separators
    product_a_row = join_with_various_separators(product_a_values, separators)
    product_b_row = join_with_various_separators(product_b_values, separators)

    # Format the rows for Product A and B
    # product_a_row = " | ".join(product_a_values)
    # product_b_row = " | ".join(product_b_values)

    # Combine header, separator, and rows into the final query
    query = (
        f"{category} Comparison:\n{header_line}\n{separator}\n"
        f"{product_a_row}\n{product_b_row}\n"
    )

    return query


def get_prompt_html_format(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    """
    Construct a query from the candidate pair data for entity matching, formatting the details in an HTML table layout.
    The column headers and rows are generated dynamically based on the provided columns list.

    Args:
    - candidate_pair (pd.Series): A single row from the candidate pairs DataFrame, representing a pair of entities to compare.
    - columns (list of str): List of column names to include in the table, without '_x' and '_y' suffixes.

    Returns:
    - str: A query string formatted for prompting GPT, describing both entities in an HTML table layout with dynamic headers.
    """

    # Create the header dynamically
    headers = [col.capitalize() for col in columns]
    header_line = " | ".join(headers)

    # Initialize rows for Product A and B
    product_a_values = ["A"]
    product_b_values = ["B"]

    # Collect values for each column
    for col in columns:
        value_a = str(candidate_pair.get(f"{col}_x", "N/A"))
        value_b = str(candidate_pair.get(f"{col}_y", "N/A"))
        product_a_values.append(value_a)
        product_b_values.append(value_b)

    # Format the rows for Product A and B
    product_a_row = " | ".join(product_a_values)
    product_b_row = " | ".join(product_b_values)

    # Combine header, separator, and rows into the final query
    query = (
        f"<table>\n<tr><th>{header_line}</th></tr>\n"
        f"<tr><td>{product_a_row}</td></tr>\n<tr><td>{product_b_row}</td></tr>\n"
    )

    return query


def get_prompt_html_format(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    """
    Construct a query from the candidate pair data for entity matching, formatting the details in an HTML table layout.
    The column headers and rows are generated dynamically based on the provided columns list.

    Args:
    - candidate_pair (pd.Series): A single row from the candidate pairs DataFrame, representing a pair of entities to compare.
    - columns (list of str): List of column names to include in the table, without '_x' and '_y' suffixes.

    Returns:
    - str: A query string formatted for prompting GPT, describing both entities in an HTML table layout with dynamic headers.
    """

    # Start the HTML table and add a row for the headers
    html_str = f"<table>\n<tr>"
    for col in columns:
        html_str += f"<th>{col.capitalize()}</th>"
    html_str += "</tr>\n"

    # Add the rows for Product A and B
    product_a_values = [str(candidate_pair.get(f"{col}_x", "N/A")) for col in columns]
    product_b_values = [str(candidate_pair.get(f"{col}_y", "N/A")) for col in columns]

    # Constructing rows for Product A and B
    html_str += (
        "<tr>" + "".join(f"<td>{value}</td>" for value in product_a_values) + "</tr>\n"
    )
    html_str += (
        "<tr" + "".join(f"<td>{value}</td>" for value in product_b_values) + "</tr>\n"
    )
    html_str += "</table>"

    # Construct the final query

    return html_str


def get_prompt_indented_delimiters(candidate_pair, columns, category="Product"):
    # Initialize strings for Product A and B details
    product_a_details = f"{category} A"
    product_b_details = f"{category} B"

    # Iterate through each specified column
    for col in columns:
        col_x = col + "_x"
        col_y = col + "_y"

        # Check and append details for Product A
        if col_x in candidate_pair:
            product_a_details += f"\n  - {col}: ```{candidate_pair[col_x]}```"

        # Check and append details for Product B
        if col_y in candidate_pair:
            product_b_details += f"\n  - {col}: ```{candidate_pair[col_y]}```"

    # Construct the final query
    query = f"{product_a_details}\n\n{product_b_details}"

    return query


def get_prompt_json_format(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    """
    Construct a query from the candidate pair data for entity matching, formatting the details in a JSON-like layout.
    This format will create a single JSON structure that contains separate sections for Product A and Product B,
    similar to the structure used in the XML format function.

    Args:
    - candidate_pair (pd.Series): A single row from the candidate pairs DataFrame, representing a pair of entities to compare.
    - columns (list of str): List of column names to include, without '_x' and '_y' suffixes.
    - category (str): The category of items being compared (default: "Product").

    Returns:
    - str: A query string formatted for prompting GPT, describing entities A and B in a JSON-like layout.
    """

    # Initialize the dictionary to hold both Product A and B details
    products_json = {f"{category} A": {}, f"{category} B": {}}

    # Iterate through each specified column to add attributes for Product A and B
    for col in columns:
        col_x = col + "_x"
        col_y = col + "_y"

        # Collect values for each column for both products
        value_a = str(candidate_pair.get(col_x, "N/A"))
        value_b = str(candidate_pair.get(col_y, "N/A"))

        # Add the key-value pairs to the dictionary for Product A and B
        products_json[f"{category} A"][col.capitalize()] = value_a
        products_json[f"{category} B"][col.capitalize()] = value_b

    # Convert the dictionary to a JSON-like string format
    query = json.dumps(products_json, indent=2)

    return query


def get_prompt_xml_format(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    """
    Construct a query from the candidate pair data for entity matching, formatting the details in an XML-like layout.
    This version separates Product A and Product B into distinct sections within the XML structure,
    similar to the separation seen in the JSON format function.

    Args:
    - candidate_pair (pd.Series): A single row from the candidate pairs DataFrame, representing a pair of entities to compare.
    - columns (list of str): List of column names to include in the table, without '_x' and '_y' suffixes.

    Returns:
    - str: A query string formatted for prompting GPT, describing both entities in an XML-like layout with dynamic headers.
    """

    # Initialize the XML-like string with the category name and distinct sections for Product A and Product B
    xml_str = f"<{category}Comparison>\n"

    # Adding separate sections for Product A and Product B
    xml_str += f"  <{category}A>\n"
    for col in columns:
        col_x = col + "_x"
        value_a = str(candidate_pair.get(col_x, "N/A"))
        xml_str += f"    <{col.capitalize()}>{value_a}</{col.capitalize()}>\n"
    xml_str += f"  </{category}A>\n"

    xml_str += f"  <{category}B>\n"
    for col in columns:
        col_y = col + "_y"
        value_b = str(candidate_pair.get(col_y, "N/A"))
        xml_str += f"    <{col.capitalize()}>{value_b}</{col.capitalize()}>\n"
    xml_str += f"  </{category}B>\n"

    # Add the closing tag for the category comparison
    xml_str += f"</{category}Comparison>"

    return xml_str


# Yaml format with same structure as XML
def get_prompt_yaml_format(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    """
    Construct a query from the candidate pair data for entity matching, formatting the details in a YAML-like layout.
    This version separates Product A and Product B into distinct sections within the YAML structure,
    similar to the separation seen in the XML format function.

    Args:
    - candidate_pair (pd.Series): A single row from the candidate pairs DataFrame, representing a pair of entities to compare.
    - columns (list of str): List of column names to include in the table, without '_x' and '_y' suffixes.

    Returns:
    - str: A query string formatted for prompting GPT, describing both entities in a YAML-like layout with dynamic headers.
    """

    # Initialize the YAML-like string with the category name and distinct sections for Product A and Product B
    yaml_str = f"{category} Comparison:\n"

    # Adding separate sections for Product A and Product B
    yaml_str += f"  {category}A:\n"
    for col in columns:
        col_x = col + "_x"
        value_a = str(candidate_pair.get(col_x, "N/A"))
        yaml_str += f"    {col.capitalize()}: {value_a}\n"

    yaml_str += f"  {category}B:\n"
    for col in columns:
        col_y = col + "_y"
        value_b = str(candidate_pair.get(col_y, "N/A"))
        yaml_str += f"    {col.capitalize()}: {value_b}\n"

    return yaml_str


def get_prompt_step_format(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    """
    Construct a query from the candidate pair data for entity matching, formatting the details in a step-by-step layout.
    This function separates the data into steps, presenting each attribute comparison for Product A and Product B.

    Args:
    - candidate_pair (pd.Series): A single row from the candidate pairs DataFrame, representing a pair of entities to compare.
    - columns (list of str): List of column names to include in the table, without '_x' and '_y' suffixes.

    Returns:
    - str: A query string formatted for prompting GPT, describing both entities in a step-by-step layout with dynamic headers.
    """

    # Initialize the string to format the output
    prompt_str = ""

    # Loop through each column, creating a step for each attribute
    step_number = 1
    for col in columns:
        col_x = col + "_x"
        col_y = col + "_y"
        value_a = str(candidate_pair.get(col_x, "N/A"))
        value_b = str(candidate_pair.get(col_y, "N/A"))

        # Adjusting for NaN values which can appear in pandas Series
        if pd.isna(value_a):
            value_a = "<nan>"
        if pd.isna(value_b):
            value_b = "<nan>"

        # Append each step to the prompt string
        prompt_str += f"Step {step_number}: {col.capitalize()}:\n"
        prompt_str += f"{category} A is {col.lower()} '{value_a}'\n"
        prompt_str += f"{category} B is {col.lower()} '{value_b}'\n\n"
        step_number += 1

    return prompt_str


# Angle brackets format with same structure as YAML, but the angle brackets surround the values
def get_prompt_angle_brackets(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    """
    Construct a query from the candidate pair data for entity matching, formatting the details in a YAML-like layout.
    This version separates Product A and Product B into distinct sections within the YAML structure,
    similar to the separation seen in the XML format function.

    Args:
    - candidate_pair (pd.Series): A single row from the candidate pairs DataFrame, representing a pair of entities to compare.
    - columns (list of str): List of column names to include in the table, without '_x' and '_y' suffixes.

    Returns:
    - str: A query string formatted for prompting GPT, describing both entities in a YAML-like layout with dynamic headers.
    """

    # Initialize the YAML-like string with the category name and distinct sections for Product A and Product B
    yaml_str = f"{category} Comparison:\n"

    # Adding separate sections for Product A and Product B
    yaml_str += f"  {category}A:\n"
    for col in columns:
        col_x = col + "_x"
        value_a = str(candidate_pair.get(col_x, "N/A"))
        yaml_str += f"    {col.capitalize()}: <{value_a}>\n"

    yaml_str += f"  {category}B:\n"
    for col in columns:
        col_y = col + "_y"
        value_b = str(candidate_pair.get(col_y, "N/A"))
        yaml_str += f"    {col.capitalize()}: <{value_b}>\n"

    return yaml_str


# Yaml format with same structure as XML
def get_prompt_gpt_format(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    """ """

    prompt_str = ""
    # Adding separate sections for Product A and Product B
    prompt_str += f"1.  **{category} A**:\n"
    for col in columns:
        col_x = col + "_x"
        value_a = str(candidate_pair.get(col_x, "N/A"))
        prompt_str += f"    - {col.capitalize()}: {value_a}\n"

    prompt_str += f"\n"
    prompt_str += f"2.  **{category} B**:\n"
    for col in columns:
        col_y = col + "_y"
        value_b = str(candidate_pair.get(col_y, "N/A"))
        prompt_str += f"    - {col.capitalize()}: {value_b}\n"

    return prompt_str


def get_prompt_step_short_format(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    """
    Construct a query from the candidate pair data for entity matching, formatting the details in a step-by-step layout.
    This function separates the data into steps, presenting each attribute comparison for Product A and Product B.

    Args:
    - candidate_pair (pd.Series): A single row from the candidate pairs DataFrame, representing a pair of entities to compare.
    - columns (list of str): List of column names to include in the table, without '_x' and '_y' suffixes.

    Returns:
    - str: A query string formatted for prompting GPT, describing both entities in a step-by-step layout with dynamic headers.
    """

    # Initialize the string to format the output
    prompt_str = ""

    prompt_str += f"Compare the values of each attribute. \n\n"

    # Loop through each column, creating a step for each attribute
    step_number = 1
    for col in columns:
        col_x = col + "_x"
        col_y = col + "_y"
        value_a = str(candidate_pair.get(col_x, "N/A"))
        value_b = str(candidate_pair.get(col_y, "N/A"))

        # Adjusting for NaN values which can appear in pandas Series
        if pd.isna(value_a):
            value_a = "<nan>"
        if pd.isna(value_b):
            value_b = "<nan>"

        # Append each step to the prompt string
        prompt_str += f"{step_number}: {col}:\n"
        prompt_str += f"    - {category} A {col} has value '{value_a}'\n"
        prompt_str += f"    - {category} B {col} has value '{value_b}'\n\n"
        step_number += 1

    return prompt_str


def get_prompt_step_format(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    """
    Construct a query from the candidate pair data for entity matching, formatting the details in a step-by-step layout.
    This function separates the data into steps, presenting each attribute comparison for Product A and Product B.

    Args:
    - candidate_pair (pd.Series): A single row from the candidate pairs DataFrame, representing a pair of entities to compare.
    - columns (list of str): List of column names to include in the table, without '_x' and '_y' suffixes.

    Returns:
    - str: A query string formatted for prompting GPT, describing both entities in a step-by-step layout with dynamic headers.
    """

    # Initialize the string to format the output
    prompt_str = ""

    # Loop through each column, creating a step for each attribute
    step_number = 1
    for col in columns:
        col_x = col + "_x"
        col_y = col + "_y"
        value_a = str(candidate_pair.get(col_x, "N/A"))
        value_b = str(candidate_pair.get(col_y, "N/A"))

        # Adjusting for NaN values which can appear in pandas Series
        if pd.isna(value_a):
            value_a = "<nan>"
        if pd.isna(value_b):
            value_b = "<nan>"

        # Append each step to the prompt string
        prompt_str += f"{step_number}: Compare the values of the attribute {col}:\n"
        prompt_str += f"    - {category} A {col} has value '{value_a}'\n"
        prompt_str += f"    - {category} B {col} has value '{value_b}'\n\n"
        step_number += 1

    return prompt_str


step_natural_format_example = """
Your task is to determine if Entity A and Entity B are the same.

1: Song_name:
		- Entity A the song is named 'The Woodland Realm ( Extended Version )'
		- Entity B the song is named  'The High Fells ( Extended Version )'

2: Artist_name:
		- Entity A the artist is named 'Howard Shore'
		- Entity B the artist is named 'Howard Shore'
"""


def get_prompt_step_natural_format(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    """
    Construct a query from the candidate pair data for entity matching, formatting the details in a step-by-step layout.
    This function separates the data into steps, presenting each attribute comparison for Product A and Product B.

    Args:
    - candidate_pair (pd.Series): A single row from the candidate pairs DataFrame, representing a pair of entities to compare.
    - columns (list of str): List of column names to include in the table, without '_x' and '_y' suffixes.

    Returns:
    - str: A query string formatted for prompting GPT, describing both entities in a step-by-step layout with dynamic headers.
    """

    # Define a template for formatting each attribute in natural language
    attribute_templates = {
        "title": "titled '{}'",
        "name": "named '{}'",
        "category": "in the category '{}'",
        "brand": "from the brand '{}'",
        "modelno": "with model number '{}'",
        "price": "priced at ${}",
        "manufacturer": "manufactured by '{}'",
        "addr": "located at '{}'",
        "city": "in '{}'",
        "phone": "contactable via '{}'",
        "type": "categorized as '{}'",
        "class": "with a class rating of '{}'",
        "description": "described as '{}'",
        "content": "with content '{}'",
    }

    # Helper function to format attribute text
    def format_attribute(col, value):
        template = attribute_templates.get(col.lower(), "{}")
        return template.format(value)

    # Initialize the string to format the output
    prompt_str = ""

    # Loop through each column, creating a step for each attribute
    step_number = 1
    for col in columns:
        col_x = col + "_x"
        col_y = col + "_y"
        value_a = str(candidate_pair.get(col_x, "N/A"))
        value_b = str(candidate_pair.get(col_y, "N/A"))

        # Adjusting for NaN values which can appear in pandas Series
        if pd.isna(value_a):
            value_a = "<nan>"
        if pd.isna(value_b):
            value_b = "<nan>"

        # Append each step to the prompt string
        prompt_str += f"{step_number}: {col}:\n"
        prompt_str += f"    - {category} A is {format_attribute(col, value_a)}\n"
        prompt_str += f"    - {category} B is {format_attribute(col, value_b)}\n\n"
        step_number += 1

    return prompt_str


# Normal natural prompt
def get_prompt_step_natural_format(
    candidate_pair: pd.Series, columns: List[str], category: str = "Product"
):
    """
    Construct a query from the candidate pair data for entity matching, formatting the details in a natural language description.

    Args:
    - candidate_pair (pd.Series): A single row from the candidate pairs DataFrame, representing a pair of entities to compare.
    - columns (list of str): List of column names to include in the prompt, formatted in a natural language style.

    Returns:
    - str: A query string formatted for prompting GPT, describing both entities in natural language.
    """
    # Start the description for Product A and B
    # NOTE: Change it to A is a {category} and B is a {category}?
    # Initialize the string to format the output
    prompt_str = ""

    # Define a template for formatting each attribute in natural language
    attribute_templates = {
        "title": "is titled '{}'",
        "name": "is named '{}'",
        "category": "has the category '{}'",
        "brand": "is from the brand '{}'",
        "modelno": "has model number '{}'",
        "price": "is priced at {}",
        "genre": "has genre '{}'",
        "manufacturer": "is manufactured by '{}'",
        "artist_name": "the artist is named '{}'",
        "song_name": "the song is named '{}'",
        "album_name": "the album is named '{}'",
        "copyright": "has copyright '{}'",
        "addr": "is located at '{}'",
        "city": "in the city '{}'",
        "phone": "is contactable via '{}'",
        "type": "is categorized as '{}'",
        "class": "has a class rating of '{}'",
        "released": "was released on '{}'",
        "time": "has a duration of '{}'",
        "description": "is described as '{}'",
        "content": "has the content '{}'",
    }

    # Helper function to format attribute text
    def format_attribute(col, value):
        template = attribute_templates.get(col.lower(), "{}")
        return template.format(value)

    # Loop through each column, creating a step for each attribute
    step_number = 1
    # Iterate through each specified column
    for col in columns:
        col_x = col + "_x"
        col_y = col + "_y"

        prompt_str += f"{step_number}: {col}:\n"

        # Format and append details for Product A
        value_a = candidate_pair.get(col_x, "")
        if pd.notna(value_a) and value_a != "":
            prompt_str += f"    - Entity A {format_attribute(col, value_a)}\n"

        # Format and append details for Product B
        value_b = candidate_pair.get(col_y, "")
        if pd.notna(value_b) and value_b != "":
            prompt_str += f"    - Entity B {format_attribute(col, value_b)}\n\n"

        step_number += 1

    # Construct the final query
    query = prompt_str

    return query
