import argparse
import logging
import ssl

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import XLNetForSequenceClassification, XLNetConfig, XLNetTokenizerFast
from datasets import concatenate_datasets

from src.finetune import SemanticFragmentsFinetuning
from src.nli_datasets import SemanticFragmentDataset

from datasets.utils.logging import set_verbosity_error

set_verbosity_error()

ssl._create_default_https_context = ssl._create_unverified_context


def setup_logger():
    log_format = '%(asctime)s - %(name)s:%(levelname)s - %(message)s'
    logging.basicConfig(filename='train_details.log', level=logging.INFO, format=log_format, datefmt='%H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def load_transformer_model(model_name: str = "xlnet-base-cased", base_model_name: str = "xlnet-base-cased"):
    config = XLNetConfig.from_pretrained(base_model_name, num_labels=3)
    tokenizer = XLNetTokenizerFast.from_pretrained(base_model_name, config=config, do_lower_case=True)
    model = XLNetForSequenceClassification.from_pretrained(model_name, config=config)
    if torch.cuda.is_available():
        model = model.to('cuda')
    return model, tokenizer


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(args):
    logging.getLogger().info("Load model and tokenizer")
    model, tokenizer = load_transformer_model(model_name=args.nli_base_model)
    
    logging.getLogger().info("Loading datasets...")
    nli_dataset = SemanticFragmentDataset(tokenizer=tokenizer)
    
    semantic_fragments = ['quantifier', 'negation', 'monotonicity_simple', 'monotonicity_hard',
                          'counting', 'conditional', 'comparative', 'boolean']
    all_fragments_datasets = []
    all_validation_sets = {}
    for fragment in semantic_fragments:
        train_file = f"{args.data_dir}/{fragment}/train/challenge_train.tsv"
        train_dataset = nli_dataset.get_fragment_dataset(train_file, threads=args.threads)
        all_fragments_datasets.append(train_dataset)
        val_file = f"{args.data_dir}/{fragment}/train/challenge_dev.tsv"
        val_dataset = nli_dataset.get_fragment_dataset(val_file, threads=args.threads)
        all_validation_sets[fragment] = DataLoader(val_dataset, batch_size=args.val_batch_size)
    
    train_dataset = concatenate_datasets(all_fragments_datasets)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    finetuning = SemanticFragmentsFinetuning(output_model_dir="semantic_fragment_models",
                                             lr=args.lr,
                                             epochs=args.epochs,
                                             gradient_accumulation_steps=args.gradient_accumulation_steps,
                                             val_steps=args.val_steps,
                                             val_dataloaders=all_validation_sets)
    logging.getLogger().info("Train - Semantic Fragments")
    finetuning.train(model, train_dataloader, initial_best_score=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune NLI Model')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Train batch size')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Validation batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Gradient Accumulation Steps')
    parser.add_argument('--val_steps', type=int, default=2000, help='Number of train steps to do validation')
    parser.add_argument('--epochs', type=int, default=4, help='Train epochs')
    parser.add_argument('--threads', type=int, default=4, help='Threads')
    parser.add_argument('--seed', type=int, default=42, help='SEED')
    parser.add_argument('--data_dir', type=str, help='Semantic Fragment dataset directory', required=True)
    parser.add_argument('--nli_base_model', type=str, help='NLI pretrained base model', required=True)
    
    args = parser.parse_args()
    setup_logger()
    set_seed(args.seed)
    logging.getLogger().info(args)
    main(args)
