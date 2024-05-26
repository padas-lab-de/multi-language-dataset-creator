# Multi-Language Dataset Creator

This Python application allows you to create a balanced multi-language dataset from the Wiki40B dataset. The dataset can include multiple languages, and you can ensure that all language datasets have a similar size in terms of token numbers within a specified range of difference. The application also supports forcing samples from different languages to be from the same wiki articles.

## Features

- Choose multiple languages for the dataset.
- Force samples from different languages to be from the same articles.
- Ensure all datasets have a similar size in terms of token numbers.
- Export the datasets to JSON files.

## Requirements

- Python 3.6 or higher
- Required Python packages:
  - `argparse`
  - `json`
  - `pickle`
  - `random`
  - `tensorflow_datasets`
  - `tqdm`
  - `tiktoken`
  - `os`

You can install the required packages using pip:

```bash
pip install tensorflow_datasets tqdm tiktoken
```

## Usage

### Command Line Arguments

- `--langs`: Comma-separated list of languages (default: `en,fr,de`).
- `--common_articles`: Force samples from different languages to be from the same articles (default: `False`).
- `--output_dir`: Output directory to save datasets (default: `.`).
- `--max_diff_percent`: Maximum allowed percentage difference in token counts (default: `5.0`).

### Example Commands

1. Create a dataset with English, French, and German, with common articles, enforcing a token limit with a 5% difference, and save the output to `./output`:
    ```sh
    python multi_language_dataset_creator.py --langs en,fr,de --common_articles --output_dir ./output --max_diff_percent 5.0
    ```

2. Create a dataset with only English and French, without forcing common articles, and save the output to the current directory:
    ```sh
    python multi_language_dataset_creator.py --langs en,fr --max_diff_percent 5.0
    ```

## TODO

- [ ] Add functionality to support programming languages in the dataset.
  - [ ] Integrate with datasets like CodeParrot or other code repositories.
  - [ ] Allow the user to specify programming languages in addition to natural languages.
  - [ ] Ensure the token limit enforcement works for both natural languages and programming languages.
  - [ ] Provide an option to balance the dataset size between natural and programming languages.
- [ ] Add functionality for training a tokenizer on the generated datasets.
  - [ ] Allow the user to specify tokenizer training parameters (e.g. vocab_size).
  - [ ] Save the trained tokenizer to the specified output directory.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [TensorFlow Datasets](https://www.tensorflow.org/datasets) for providing the Wiki40B dataset.
- [tiktoken](https://github.com/openai/tiktoken) for token counting.
