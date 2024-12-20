# NAIC Entity Matching Demonstrator

# Applying Prompt Engineering for Entity Matching
This repository contains the code for experimenting with various prompt engineering techniques for entity matching. 

## Setup
1. Clone the repository
2. Install the required packages using the following command:
```bash
    pip install -r requirements.txt
```
3. Download the datasets 
```bash
    python utils/download_datasets.py
```
4. Ensure that OPENAI_API_KEY is set in the environment variables. 

## Notebook 
The notebook `Demonstrator.ipynb` contains examples for how to run some experiments and visualize the results.

## Running experiments
To run a zero-shot experiment on the abt-buy dataset using the natural prompt, 200 pairs, GPT-4o as the language model, context and "Be lenient in your judgement" added, run the following command:
```bash
    python main.py -d abt-buy -pf natural -n 200 -k 0 -llm gpt-4o -imp leneint -ctx
```

To run a few-shot experiment on the dblp_gs_dirty dataset using the tabular prompt, all pairs (if the number is larger than the dataset, the entire dataset will be used), 10 examples in the prompt with a 30/70 split of positive and negative examples respectively, GPT-3 as the language model, basic prompt format, and no sublte context, the following command can be used:
```bash
    python main.py -d dblp_gs_dirty -pf tabular -n 10000 -k 10 -llm gpt-3 -imp basic -ctp '(P,N)' -pn "30/70"
```

To get a list of all the available options, run the following command:
```bash
    python main.py --help
```


