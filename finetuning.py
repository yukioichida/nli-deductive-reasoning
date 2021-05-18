import argparse
import logging
import math
import ssl

import torch
from tqdm import tqdm
from transformers import AdamW, get_scheduler
from transformers import XLNetForSequenceClassification, XLNetConfig, XLNetTokenizer
from datasets.metric import Metric

from src.nli_datasets import NLIDatasets

ssl._create_default_https_context = ssl._create_unverified_context


def validate(model, val_dataloader, metric) -> Metric:
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            if torch.cuda.is_available():
                batch = {key: tensor.to('cuda') for key, tensor in batch.items()}
            outputs = model(attention_mask=batch['attention_mask'],
                            input_ids=batch['input_ids'],
                            token_type_ids=batch['token_type_ids'],
                            labels=batch['label'])
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(predictions=predictions, references=batch["label"])
        return metric.compute()


def get_optimizers(model: torch.nn.Module, lr: float, train_cycles: int, gradient_accumulation_steps: int):
    num_update_steps_per_epoch = math.ceil(train_cycles / gradient_accumulation_steps)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    # Optimizer -  Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps,
    )
    return optimizer, lr_scheduler


def get_logger():
    log_format = '%(asctime)s - %(message)s'
    logging.basicConfig(filename='train_details.log', level=logging.INFO, format=log_format, datefmt='%H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    return logging.getLogger(__name__)


def train(lr: float, train_batch_size: int, val_batch_size: int, gradient_accumulation_steps: int, val_step: int,
          epochs: int, threads: int):
    model_name = "xlnet-base-cased"
    config = XLNetConfig.from_pretrained(model_name, num_labels=3)
    tokenizer = XLNetTokenizer.from_pretrained(model_name, config=config)
    model = XLNetForSequenceClassification.from_pretrained(model_name, config=config)
    if torch.cuda.is_available():
        model = model.to('cuda')
    
    log = get_logger()
    log.info('Loading dataset...')
    nli_dataset = NLIDatasets(tokenizer=tokenizer, threads=threads)
    train_loader = nli_dataset.get_train_dataloader(train_batch_size=train_batch_size)
    val_m_loader, val_mm_loader = nli_dataset.get_mnli_dev_dataloaders(val_batch_size=val_batch_size)
    
    metric = nli_dataset.get_metric()
    train_loader_len = len(train_loader)
    optimizer, lr_scheduler = get_optimizers(model=model, lr=lr, train_cycles=train_loader_len,
                                             gradient_accumulation_steps=gradient_accumulation_steps)
    
    # Train
    log.info("Train...")
    for epoch in range(epochs):
        for step, batch in tqdm(enumerate(train_loader), total=train_loader_len):
            model.train()
            if torch.cuda.is_available():
                batch = {key: tensor.to('cuda') for key, tensor in batch.items()}
            
            outputs = model(attention_mask=batch['attention_mask'],
                            input_ids=batch['input_ids'],
                            token_type_ids=batch['token_type_ids'],
                            labels=batch['label'])
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0 or step == train_loader_len - 1:
                # Propagation
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if (step + 1) % val_step == 0:
                val_acc = validate(model, val_m_loader, metric)['accuracy']
                log.info(f"{epoch} - {step} - Val acc matched: {val_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune NLI Model')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Train batch size')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Validation batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Gradient Accumulation Steps')
    parser.add_argument('--val_step', type=int, default=2000, help='Number of train steps to do validation')
    parser.add_argument('--epochs', type=int, default=3, help='Train epochs')
    parser.add_argument('--threads', type=int, default=4, help='Threads')
    
    args = parser.parse_args()
    train(lr=args.lr,
          train_batch_size=args.train_batch_size,
          val_batch_size=args.val_batch_size,
          gradient_accumulation_steps=args.gradient_accumulation_steps,
          val_step=args.val_step,
          epochs=args.epochs,
          threads=args.threads)
