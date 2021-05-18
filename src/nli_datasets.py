from datasets import load_dataset, load_metric, concatenate_datasets, Metric, Dataset
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader

# Disable tqdm from datasets
from datasets.utils.logging import set_verbosity_error

set_verbosity_error()


class NLIDatasets:
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 128, threads: int = 4):
        self.metric = load_metric('accuracy')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mnli_dataset = self._load_nli_datasets(load_dataset('glue', 'mnli'), thread=threads).remove_columns('idx')
        self.snli_dataset = self._load_nli_datasets(load_dataset('snli'), thread=threads)
    
    def get_metric(self) -> Metric:
        return self.metric
    
    def _load_nli_datasets(self, dataset: Dataset, thread: int = 4) -> Dataset:
        loaded_dataset = dataset.filter(lambda x: x["label"] != -1).map(
            lambda row: self.tokenizer(row['premise'],
                                       row['hypothesis'],
                                       truncation=True,
                                       padding='max_length',
                                       max_length=self.max_length),
            batched=True,
            num_proc=thread)
        loaded_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        return loaded_dataset
    
    def get_train_dataloader(self, train_batch_size: int) -> DataLoader:
        train_mnli_dataset = self.mnli_dataset['train']
        train_snli_dataset = self.snli_dataset['train']
        print(f"MNLI Train Set length: {len(train_mnli_dataset)}")
        print(f"SNLI Train Set length: {len(train_snli_dataset)}")
        train_dataset = concatenate_datasets([train_mnli_dataset, train_snli_dataset])
        print(f"Total train length {len(train_dataset)}")
        print(set(train_dataset['label']))
        return DataLoader(train_dataset, batch_size=train_batch_size)
    
    def get_mnli_dev_dataloaders(self, val_batch_size: int) -> (DataLoader, DataLoader):
        val_matched = self.mnli_dataset['validation_matched']
        val_mismatched = self.mnli_dataset['validation_mismatched']
        return DataLoader(val_matched, batch_size=val_batch_size), DataLoader(val_mismatched, batch_size=val_batch_size)
    
    def get_snli_dev_test_dataloaders(self, val_batch_size: int) -> (DataLoader, DataLoader):
        snli_val = self.snli_dataset['validation']
        sni_test = self.snli_dataset['test']
        return DataLoader(snli_val, batch_size=val_batch_size), DataLoader(sni_test, batch_size=val_batch_size)
