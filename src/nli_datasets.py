from datasets import load_dataset, load_metric, concatenate_datasets, Metric, Dataset
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader

# Disable tqdm from datasets
from datasets.utils.logging import set_verbosity_error

set_verbosity_error()


class NLIDatasets:
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 128):
        self.metric = load_metric('accuracy')
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def get_metric(self) -> Metric:
        return self.metric
    
    def _load_nli_datasets(self, dataset: Dataset, threads: int = 4) -> Dataset:
        loaded_dataset = dataset.filter(lambda x: x["label"] != -1).map(
            lambda row: self.tokenizer(row['premise'],
                                       row['hypothesis'],
                                       truncation=True,
                                       padding='max_length',
                                       max_length=self.max_length),
            batched=True,
            num_proc=threads)
        loaded_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        return loaded_dataset
    
    def get_train_dataloader(self, train_batch_size: int, threads: int = 4) -> DataLoader:
        train_mnli_dataset = load_dataset('glue', 'mnli', split='train').remove_columns('idx')
        train_mnli_dataset = self._load_nli_datasets(train_mnli_dataset, threads=threads)
        train_snli_dataset = load_dataset('snli', split='train')
        train_snli_dataset = self._load_nli_datasets(train_snli_dataset, threads=threads)
        print(f"MNLI Train Set length: {len(train_mnli_dataset)}")
        print(f"SNLI Train Set length: {len(train_snli_dataset)}")
        train_dataset = concatenate_datasets([train_mnli_dataset, train_snli_dataset])
        print(f"Total train length {len(train_dataset)}")
        return DataLoader(train_dataset, batch_size=train_batch_size)
    
    def get_mnli_dev_dataloaders(self, val_batch_size: int, threads: int = 4) -> (DataLoader, DataLoader):
        val_m, val_mm = load_dataset('glue', 'mnli',
                                     split=['validation_matched', 'validation_mismatched'])
        val_m = self._load_nli_datasets(val_m.remove_columns('idx'), threads=threads)
        val_mm = self._load_nli_datasets(val_mm.remove_columns('idx'), threads=threads)
        return DataLoader(val_m, batch_size=val_batch_size), DataLoader(val_mm, batch_size=val_batch_size)
