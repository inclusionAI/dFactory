import json
import os

from datasets import load_dataset

def format_gsm8k_example_to_messages(example):
    """
    Converts a single example from the gsm8k dataset into the messages format.

    Args:
        example (dict): A single sample from the dataset, containing 'question' and 'answer' keys.

    Returns:
        dict: A dictionary containing a 'messages' key, whose value is a list
              of dictionaries representing the user and assistant dialogue.
    """
    # 'question' corresponds to the 'user' role
    # 'answer' corresponds to the 'assistant' role
    return {
        "messages": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]}
        ]
    }

def main():
    # 1. Load the gsm8k dataset (main config) from the Hugging Face Hub
    print("Loading gsm8k (main) dataset...")
    # The 'main' configuration contains clean question-answer pairs
    raw_dataset = load_dataset("openai/gsm8k", "main")
    print("Dataset loaded successfully.")
    print(f"Original dataset structure: {raw_dataset}")
    print(f"First example from the original training set: {raw_dataset['train'][0]}")

    # 2. Use the .map() method to convert the dataset to the messages format
    # remove_columns will drop the old 'question' and 'answer' columns, keeping only the new 'messages' column
    print("\nConverting dataset to messages format...")
    transformed_dataset = raw_dataset.map(
        format_gsm8k_example_to_messages,
        remove_columns=["question", "answer"]
    )
    print("Conversion complete.")
    print(f"Transformed dataset structure: {transformed_dataset}")
    print(f"First example from the transformed training set: {transformed_dataset['train'][0]}")

    # 3. Save the processed dataset to JSONL files
    print("\nSaving to JSONL files...")
    # Ensure the output directory exists
    output_dir = "./gsm8k_datasets"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each split of the dataset (e.g., 'train' and 'test')
    for split, dataset_split in transformed_dataset.items():
        output_filename = os.path.join(output_dir, f"gsm8k_{split}.jsonl")

        with open(output_filename, 'w', encoding='utf-8') as f:
            for record in dataset_split:
                # Convert each dictionary to a JSON string and write it to the file, followed by a newline
                json_string = json.dumps(record, ensure_ascii=False)
                f.write(json_string + '\n')

        print(f"Saved the '{split}' split to: {output_filename}")

if __name__ == "__main__":
    main()
