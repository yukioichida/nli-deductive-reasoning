import argparse
import logging
import ssl

import numpy as np
import torch
from transformers import XLNetForSequenceClassification, XLNetConfig, XLNetTokenizerFast

from src.finetune import MNLISNLIFinetuning
from src.nli_datasets import DefaultNLIDataset

ssl._create_default_https_context = ssl._create_unverified_context


def setup_logger():
    log_format = '%(asctime)s - %(message)s'
    logging.basicConfig(filename='train_details.log', level=logging.INFO, format=log_format, datefmt='%H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    return logging.getLogger(__name__)


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
    model, tokenizer = load_transformer_model()
    
    logging.getLogger().info("Loading datasets...")
    nli_dataset = DefaultNLIDataset(tokenizer=tokenizer)
    train_dataloader = nli_dataset.get_train_dataloader(batch_size=args.train_batch_size, threads=args.threads)
    val_matched_dataloader, val_mismatched_dataloader = nli_dataset.get_mnli_dev_dataloaders(
        batch_size=args.val_batch_size,
        threads=args.threads)
    val_snli_dataloader = nli_dataset.get_snli_val_dataloader(batch_size=args.val_batch_size, threads=args.threads)
    
    finetuning = MNLISNLIFinetuning(output_model_dir="snli_mnli_models",
                                    lr=args.lr,
                                    epochs=args.epochs,
                                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                                    val_steps=args.val_steps,
                                    val_matched_dataloader=val_matched_dataloader,
                                    val_mismatched_dataloader=val_mismatched_dataloader,
                                    val_snli_dataloader=val_snli_dataloader)
    
    logging.getLogger().info("Train")
    finetuning.train(model, train_dataloader)


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
    
    args = parser.parse_args()
    setup_logger()
    set_seed(args.seed)
    logging.getLogger().info(args)
    main(args)
