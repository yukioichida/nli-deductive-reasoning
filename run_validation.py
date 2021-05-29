import argparse
import logging
import torch
import numpy as np
from src.nli_datasets import DefaultNLIDataset
from src.finetune import validate
from transformers import XLNetForSequenceClassification, XLNetConfig, XLNetTokenizerFast


def setup_logger():
    log_format = '%(asctime)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%H:%M:%S')


def load_transformer_model(model_name: str = "xlnet-base-cased", base_model_name: str = "xlnet-base-cased"):
    config = XLNetConfig.from_pretrained(base_model_name, num_labels=3)
    tokenizer = XLNetTokenizerFast.from_pretrained(base_model_name, config=config, do_lower_case=True)
    model = XLNetForSequenceClassification.from_pretrained(model_name, config=config)
    if torch.cuda.is_available():
        model = model.to('cuda')
    return model, tokenizer


def validation(pretrained_model: str, val_batch_size: int, n_threads: int):
    log = logging.getLogger()
    model, tokenizer = load_transformer_model(model_name=pretrained_model)
    
    nli_dataset = DefaultNLIDataset(tokenizer=tokenizer)
    metric = nli_dataset.get_metric()
    
    log.info(f"Loading MNLI  validation datasets (matched/mismatched)")
    val_m_loader, val_mm_loader = nli_dataset.get_mnli_dev_dataloaders(batch_size=val_batch_size, threads=n_threads)
    log.info(f"Loading SNLI test set")
    test_snli_loader = nli_dataset.get_snli_test_dataloader(batch_size=val_batch_size, threads=n_threads)
    
    log.info("Validating on MNLI sets")
    val_matched_acc = validate(model, val_m_loader, metric)['accuracy']
    val_mismatched_acc = validate(model, val_mm_loader, metric)['accuracy']
    val_acc = (val_matched_acc + val_mismatched_acc) / 2
    log.info(f"Val acc matched/mismatched: {val_matched_acc:.4f}/{val_mismatched_acc:.4f} - Val acc avg: {val_acc:.4f}")
    
    log.info("Validating on SNLI sets")
    test_snli_acc = validate(model, test_snli_loader, metric)['accuracy']
    log.info(f"SNLI test set acc: {test_snli_acc: .4f}")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune NLI Model')
    parser.add_argument('--pretrained_model', type=str, required=True, help='Pretrained dir model')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Validation batch size')
    parser.add_argument('--n_threads', type=int, default=4, help='Threads')
    parser.add_argument('--seed', type=int, default=42, help='SEED')
    
    args = parser.parse_args()
    setup_logger()
    set_seed(args.seed)
    logging.getLogger().info(args)
    validation(pretrained_model=args.pretrained_model, val_batch_size=args.val_batch_size, n_threads=args.n_threads)
