import argparse
import json
import tensorflow_datasets as tfds
from tqdm import tqdm
import tiktoken
import os

def load_wiki40b(langs):
    """
    Load the wiki40b datasets for the specified languages.

    Args:
    langs (list): List of language codes.

    Returns:
    dict: Dictionary of datasets for each language.
    """
    print("Step 1/7: Loading Wiki40B datasets...")
    datasets = {}
    for lang in langs:
        datasets[lang] = {
            'train': tfds.load(f'wiki40b/{lang}', split='train'),
            'validation': tfds.load(f'wiki40b/{lang}', split='validation')
        }
    print("Step 1/7: Completed loading Wiki40B datasets.")
    return datasets

def extract_wikidata_ids(datasets):
    """
    Extract Wikidata IDs for both training and validation sets for each language.

    Args:
    datasets (dict): Dictionary of datasets.

    Returns:
    dict: Dictionary of Wikidata IDs for each language and split.
    """
    print("Step 2/7: Extracting Wikidata IDs...")
    wikidata_ids = {lang: {'train': set(), 'validation': set()} for lang in datasets}
    for lang, splits in datasets.items():
        for split in splits:
            for example in tfds.as_numpy(datasets[lang][split]):
                wikidata_ids[lang][split].add(example['wikidata_id'])
    print("Step 2/7: Completed extracting Wikidata IDs.")
    return wikidata_ids

def find_common_ids(wikidata_ids, langs):
    """
    Identify common Wikidata IDs across all languages for both training and validation sets.

    Args:
    wikidata_ids (dict): Dictionary of Wikidata IDs for each language and split.
    langs (list): List of language codes.

    Returns:
    dict: Dictionary of common IDs for training and validation sets.
    """
    print("Step 3/7: Finding common Wikidata IDs...")
    common_ids = {split: set.intersection(*[wikidata_ids[lang][split] for lang in langs])
                  for split in ['train', 'validation']}
    print("Step 3/7: Completed finding common Wikidata IDs.")
    return common_ids

def build_final_datasets(datasets, common_ids, langs):
    """
    Retrieve the corresponding articles for each common ID in both training and validation sets.

    Args:
    datasets (dict): Dictionary of datasets for each language.
    common_ids (dict): Dictionary of common IDs for training and validation sets.
    langs (list): List of language codes.

    Returns:
    dict: Dictionary of final datasets for training and validation sets.
    """
    print("Step 4/7: Building final datasets...")
    final_datasets = {'train': [], 'validation': []}
    dataset_hashmaps = {lang: {split: {} for split in ['train', 'validation']} for lang in langs}
    for lang in langs:
        for split in ['train', 'validation']:
            for example in tfds.as_numpy(datasets[lang][split]):
                dataset_hashmaps[lang][split][example['wikidata_id']] = example
    for split in common_ids:
        for common_id in common_ids[split]:
            article = {lang: dataset_hashmaps[lang][split][common_id] for lang in langs}
            final_datasets[split].append(article)
    print("Step 4/7: Completed building final datasets.")
    return final_datasets

def decode_bytes(data):
    """
    Decode byte strings in the dataset to regular strings.

    Args:
    data: Data to decode.

    Returns:
    Decoded data.
    """
    if isinstance(data, bytes):
        return data.decode('utf-8')
    if isinstance(data, dict):
        return {k: decode_bytes(v) for k, v in data.items()}
    if isinstance(data, list):
        return [decode_bytes(v) for v in data]
    return data

def deep_decode(obj):
    """
    Recursively decode all byte objects within a given data structure.

    Args:
    obj: The data structure to decode.

    Returns:
    The decoded data structure.
    """
    if isinstance(obj, dict):
        return {k: deep_decode(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_decode(elem) for elem in obj]
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    else:
        return obj

def export_data(final_datasets, output_dir):
    """
    Export both training and validation datasets to JSON files.

    Args:
    final_datasets (dict): Dictionary of final datasets for training and validation sets.
    output_dir (str): Output directory to save datasets.
    """
    print("Step 5/7: Exporting data to JSON files...")
    final_datasets_decoded = deep_decode(final_datasets)
    for split in final_datasets_decoded:
        with open(os.path.join(output_dir, f'wiki40b_multilang_{split}.json'), 'w', encoding='utf-8') as file:
            json.dump(final_datasets_decoded[split], file, ensure_ascii=False, indent=4)
    print("Step 5/7: Completed exporting data to JSON files.")

def calculate_token_counts(dataset):
    """
    Calculate the total token count for a dataset.

    Args:
    dataset (list): List of articles.

    Returns:
    int: Total token count.
    """
    encoder = tiktoken.get_encoding("gpt2")
    return sum(len(encoder.encode(article['text'])) for article in dataset)

def enforce_token_limit(datasets, max_diff_percent):
    """
    Enforce a token limit so that all datasets have a similar size in terms of tokens within a specified range.

    Args:
    datasets (dict): Dictionary of datasets for each split.
    max_diff_percent (float): Maximum allowed percentage difference in token counts.

    Returns:
    dict: Adjusted datasets.
    """
    print("Step 6/7: Enforcing token limits...")
    encoder = tiktoken.get_encoding("gpt2")

    # Helper function to decode text if it's in bytes
    def decode_text(article):
        for lang in article:
            if isinstance(article[lang]['text'], bytes):
                article[lang]['text'] = article[lang]['text'].decode('utf-8')
        return article

    # Calculate token counts for each language in the 'train' split
    token_counts = {lang: calculate_token_counts([decode_text(article)[lang] for article in datasets['train']]) for lang in datasets['train'][0].keys()}
    min_tokens = min(token_counts.values())
    max_tokens = min_tokens * (1 + max_diff_percent / 100)

    # Adjust training datasets to enforce token limit
    adjusted_datasets = {'train': [], 'validation': datasets['validation']}
    
    token_counts = {lang: 0 for lang in datasets['train'][0].keys()}
    for article in datasets['train']:
        article = decode_text(article)
        article_tokens = {lang: len(encoder.encode(article[lang]['text'])) for lang in article.keys()}
        
        # Check if adding this article would exceed the max token limit for any language
        if all(token_counts[lang] + article_tokens[lang] <= max_tokens for lang in article.keys()):
            adjusted_datasets['train'].append(article)
            for lang in article.keys():
                token_counts[lang] += article_tokens[lang]

    print("Step 6/7: Completed enforcing token limits.")
    return adjusted_datasets

def split_and_save_data(datasets, output_dir):
    """
    Save the training and validation datasets to JSON files.

    Args:
    datasets (dict): Dictionary of datasets for each split.
    output_dir (str): Output directory to save datasets.

    Returns:
    tuple: New training and validation data.
    """
    print("Step 7/7: Splitting and saving data...")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Deep decode to ensure all text fields are decoded
    datasets = deep_decode(datasets)

    with open(os.path.join(output_dir, 'wiki40b_multilang_train.json'), 'w', encoding='utf-8') as file:
        json.dump(datasets['train'], file, ensure_ascii=False, indent=4)
    with open(os.path.join(output_dir, 'wiki40b_multilang_validation.json'), 'w', encoding='utf-8') as file:
        json.dump(datasets['validation'], file, ensure_ascii=False, indent=4)
    print("Step 7/7: Completed splitting and saving data.")
    return datasets['train'], datasets['validation']

def main(args):
    print("Starting the dataset creation process...")
    langs = args.langs.split(',')
    output_dir = args.output_dir
    max_diff_percent = args.max_diff_percent

    datasets = load_wiki40b(langs)
    wikidata_ids = extract_wikidata_ids(datasets)
    common_ids = find_common_ids(wikidata_ids, langs)

    if args.common_articles:
        final_datasets = build_final_datasets(datasets, common_ids, langs)
    else:
        final_datasets = {lang: {split: [decode_bytes(article) for article in tfds.as_numpy(datasets[lang][split])]
                          for split in ['train', 'validation']} for lang in langs}

    adjusted_datasets = enforce_token_limit(final_datasets, max_diff_percent)
    adjusted_datasets = deep_decode(adjusted_datasets)  # Ensure all text is decoded after adjusting
    split_and_save_data(adjusted_datasets, output_dir)
    print("Dataset creation process completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a multi-language dataset.')
    parser.add_argument('--langs', type=str, default='en,fr,de', help='Comma-separated list of languages. (e.g. en,fr,de)')
    parser.add_argument('--common_articles', action='store_true', help='Force samples from different languages to be from the same articles.')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory to save datasets.')
    parser.add_argument('--max_diff_percent', type=float, default=5.0, help='Maximum allowed percentage difference in token counts.')
    args = parser.parse_args()
    main(args)
