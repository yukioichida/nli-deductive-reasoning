import abc
import logging
import logging
import math

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_scheduler
from transformers import XLNetForSequenceClassification, XLNetConfig, XLNetTokenizerFast

from datasets import Metric, load_metric


class Finetuning:
    
    def __init__(self, output_model_dir: str, **train_args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_model_dir = output_model_dir
        self.lr = train_args['lr']
        self.epochs = train_args['epochs']
        self.gradient_accumulation_steps = train_args['gradient_accumulation_steps']
        self.val_steps = train_args['val_steps']
    
    def get_optimizers(self, model: torch.nn.Module, train_cycles: int):
        num_update_steps_per_epoch = math.ceil(train_cycles / self.gradient_accumulation_steps)
        max_train_steps = self.epochs * num_update_steps_per_epoch
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        lr_scheduler = get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=max_train_steps
        )
        return optimizer, lr_scheduler
    
    @abc.abstractmethod
    def compute_model_score(self, model: Module, step: int, epoch: int) -> float:
        pass
    
    def train(self, model: Module, train_data_loader: DataLoader, initial_best_score: float = 0.866):
        train_loader_len = len(train_data_loader)
        optimizer, lr_scheduler = self.get_optimizers(model=model, train_cycles=train_loader_len)
        # Train
        best_model_score = initial_best_score
        logging.getLogger().info("Train...")
        for epoch in range(self.epochs):
            for step, batch in tqdm(enumerate(train_data_loader), total=train_loader_len):
                model.train()
                batch = {key: tensor.to(self.device) for key, tensor in batch.items()}
                
                outputs = model(attention_mask=batch['attention_mask'],
                                input_ids=batch['input_ids'],
                                token_type_ids=batch['token_type_ids'],
                                labels=batch['label'])
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % self.gradient_accumulation_steps == 0 or step == train_loader_len - 1:
                    # Propagation
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                if (step + 1) % self.val_steps == 0:
                    model_score = self.compute_model_score(model, step=step, epoch=epoch)
                    if model_score > best_model_score:
                        logging.getLogger().info(f"Saving model: Score {model_score:.4f}")
                        model.save_pretrained(f'{self.output_model_dir}/trained-model-{model_score:.4f}.ckp')
                        best_model_score = model_score
    
    def validate(self, model: Module, val_dataloader: DataLoader, metric: Metric) -> Metric:
        model.eval()
        len_dataloader = len(val_dataloader)
        with torch.no_grad():
            for step, batch in tqdm(enumerate(val_dataloader), total=len_dataloader):
                batch = {key: tensor.to(self.device) for key, tensor in batch.items()}
                outputs = model(attention_mask=batch['attention_mask'],
                                input_ids=batch['input_ids'],
                                token_type_ids=batch['token_type_ids'],
                                labels=batch['label'])
                predictions = outputs.logits.argmax(dim=-1)
                metric.add_batch(predictions=predictions, references=batch["label"])
        return metric.compute()


class MNLISNLIFinetuning(Finetuning):
    
    def __init__(self,
                 val_matched_dataloader: DataLoader,
                 val_mismatched_dataloader: DataLoader,
                 val_snli_dataloader: DataLoader,
                 **kwargs):
        super(MNLISNLIFinetuning, self).__init__(**kwargs)
        self.val_matched_dataloader = val_matched_dataloader
        self.val_mismatched_dataloader = val_mismatched_dataloader
        self.val_snli_dataloader = val_snli_dataloader
    
    def compute_model_score(self, model: torch.nn.Module, step: int, epoch: int) -> float:
        metric = load_metric('accuracy')
        val_matched_acc = self.validate(model, self.val_matched_dataloader, metric)['accuracy']
        val_mismatched_acc = self.validate(model, self.val_mismatched_dataloader, metric)['accuracy']
        val_snli_acc = self.validate(model, self.val_snli_dataloader, metric)['accuracy']
        val_acc = (val_matched_acc + val_mismatched_acc + val_snli_acc) / 3
        logging.getLogger().info(f"{epoch} - {step} - Val acc matched/mismatched/SNLI dev: "
                                 f"{val_matched_acc:.4f}/{val_mismatched_acc:.4f}/{val_snli_acc:.4f}"
                                 f"- Val acc avg: {val_acc:.4f}")
        return val_acc
