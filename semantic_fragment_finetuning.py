import argparse
import logging
import ssl

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import XLNetForSequenceClassification, XLNetConfig, XLNetTokenizerFast
from datasets import concatenate_datasets

from src.finetune import SemanticFragmentsFinetuning
from src.nli_datasets import SemanticFragmentDataset, DefaultNLIDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from datasets.utils.logging import set_verbosity_error

set_verbosity_error()

ssl._create_default_https_context = ssl._create_unverified_context


def setup_logger():
    log_format = '%(asctime)s - %(name)s:%(levelname)s - %(message)s'
    # create logger with 'spam_application'
    logger = logging.getLogger("finetuning")
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('finetuning.log')
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)


def load_transformer_model(model_name: str, base_model_name: str):
    tokenizer = AutoTokenizer.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli", do_lower_case=True)
    model = AutoModelForSequenceClassification.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
    if torch.cuda.is_available():
        model = model.to('cuda')
    return model, tokenizer


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(args):
    model_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    logging.getLogger("finetuning").info("Load model and tokenizer")
    model, tokenizer = load_transformer_model(model_name=model_name, base_model_name=model_name)
    
    logging.getLogger("finetuning").info("Loading datasets...")
    mnli_dataset = DefaultNLIDataset(tokenizer=tokenizer)
    matched_dataloader, mismatched_dataloader = mnli_dataset.get_mnli_dev_dataloaders(batch_size=args.val_batch_size,
                                                                                      threads=args.threads)
    nli_dataset = SemanticFragmentDataset(tokenizer=tokenizer, max_length=256)
    
    logical_fragments = ['quantifier', 'negation', 'counting', 'conditional', 'comparative', 'boolean']
    monotonicity_fragments = ['monotonicity_simple', 'monotonicity_hard']
    if args.only_logic:
        semantic_fragments = logical_fragments
    else:
        semantic_fragments = logical_fragments + monotonicity_fragments
    
    all_fragments_datasets = []
    all_validation_sets = {}
    for fragment in semantic_fragments:
        train_file = f"{args.data_dir}/{fragment}/train/challenge_train.tsv"
        train_dataset = nli_dataset.get_fragment_dataset(train_file, threads=args.threads)
        all_fragments_datasets.append(train_dataset)
        val_file = f"{args.data_dir}/{fragment}/train/challenge_dev.tsv"
        val_dataset = nli_dataset.get_fragment_dataset(val_file, threads=args.threads)
        all_validation_sets[fragment] = DataLoader(val_dataset, batch_size=args.val_batch_size, drop_last=False)
    
    all_validation_sets['val_mnli_matched'] = matched_dataloader
    all_validation_sets['val_mnli_mismatched'] = mismatched_dataloader
    
    train_dataset = concatenate_datasets(all_fragments_datasets)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    finetuning = SemanticFragmentsFinetuning(output_model_dir="semantic_fragment_models",
                                             lr=args.lr,
                                             epochs=args.epochs,
                                             gradient_accumulation_steps=args.gradient_accumulation_steps,
                                             val_steps=args.val_steps,
                                             val_dataloaders=all_validation_sets,
                                             save_model=args.save_model)
    logging.getLogger("finetuning").info("Train - Semantic Fragments")
    if args.validate:
        model.eval()
        with torch.no_grad():
            finetuning.compute_model_score(model)
    else:
        finetuning.train(model, train_dataloader, initial_best_score=0.91)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune NLI Model')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Train batch size')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Validation batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient Accumulation Steps')
    parser.add_argument('--val_steps', type=int, default=2000, help='Number of train steps to do validation')
    parser.add_argument('--epochs', type=int, default=4, help='Train epochs')
    parser.add_argument('--threads', type=int, default=4, help='Threads')
    parser.add_argument('--seed', type=int, default=42, help='SEED')
    parser.add_argument('--data_dir', type=str, help='Semantic Fragment dataset directory', required=True)
    parser.add_argument('--nli_base_model', type=str, help='NLI pretrained base model', required=True)
    parser.add_argument('--only_logic', action='store_true', default=False, help='Use only logic fragments')
    parser.add_argument('--validate', action='store_true', default=False, help='Validate only')
    parser.add_argument('--save_model', action='store_true', default=False, help='Save model')
    
    args = parser.parse_args()
    setup_logger()
    set_seed(args.seed)
    logging.getLogger("finetuning").info(args)
    main(args)
