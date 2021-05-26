from datasets import load_dataset, load_metric, concatenate_datasets, Metric, Dataset, Features, Value
from datasets.features import ClassLabel
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader

import pandas as pd

# Disable tqdm from datasets
from datasets.utils.logging import set_verbosity_error

set_verbosity_error()


class NLIDataset:
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 128):
        self.metric = load_metric('accuracy')
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def get_metric(self) -> Metric:
        return self.metric
    
    def _load_nli_datasets(self, dataset: Dataset, threads: int = 4) -> Dataset:
        loaded_dataset = dataset.filter(lambda x: x["label"] != -1).map(
            lambda row: self.tokenizer(row['premise'], row['hypothesis'],
                                       truncation=True, padding='max_length', max_length=self.max_length),
            batched=True,
            num_proc=threads)
        loaded_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        return loaded_dataset


class DefaultNLIDataset(NLIDataset):
    
    def get_train_dataloader(self, batch_size: int, threads: int = 4) -> DataLoader:
        train_mnli_dataset = load_dataset('glue', 'mnli', split='train').remove_columns('idx')
        train_mnli_dataset = self._load_nli_datasets(train_mnli_dataset, threads=threads)
        train_snli_dataset = self._load_nli_datasets(load_dataset('snli', split='train'), threads=threads)
        train_dataset = concatenate_datasets([train_mnli_dataset, train_snli_dataset])
        print(f"MNLI/SNLI Train samples: {len(train_mnli_dataset)}/{len(train_snli_dataset)} "
              f"- Total: {len(train_dataset)}")
        return DataLoader(train_dataset, batch_size=batch_size)
    
    def get_mnli_dev_dataloaders(self, batch_size: int, threads: int = 4) -> (DataLoader, DataLoader):
        val_m, val_mm = load_dataset('glue', 'mnli', split=['validation_matched', 'validation_mismatched'])
        val_m = self._load_nli_datasets(val_m.remove_columns('idx'), threads=threads)
        val_mm = self._load_nli_datasets(val_mm.remove_columns('idx'), threads=threads)
        return DataLoader(val_m, batch_size=batch_size), DataLoader(val_mm, batch_size=batch_size)
    
    def get_snli_test_dataloader(self, batch_size: int, threads: int = 4) -> DataLoader:
        test_snli_set = load_dataset('snli', split='test')
        test_snli_set = self._load_nli_datasets(test_snli_set, threads=threads)
        return DataLoader(test_snli_set, batch_size=batch_size)
    
    def get_snli_val_dataloader(self, batch_size: int, threads: int = 4) -> DataLoader:
        val_snli_set = load_dataset('snli', split='validation')
        val_snli_set = self._load_nli_datasets(val_snli_set, threads=threads)
        return DataLoader(val_snli_set, batch_size=batch_size)


class SemanticFragmentDataset(NLIDataset):
    
    @staticmethod
    def _convert_tsv_to_dataset(data_file: str) -> Dataset:
        df = pd.read_csv(data_file, sep='\t', names=['id', 'premise', 'hypothesis', 'label'], header=None)
        class_labels = ClassLabel(names=["ENTAILMENT", "NEUTRAL", "CONTRADICTION"], num_classes=3)
        features = Features(
            {
                "premise": Value("string"),
                "hypothesis": Value("string"),
                "label": class_labels
            })
        df['label'] = df['label'].map(lambda label: class_labels.str2int(label))
        return Dataset.from_pandas(df[['premise', 'hypothesis', 'label']], features=features)
    
    def get_file_dataloader(self, dataset_file: str, batch_size: int, threads: int = 4) -> DataLoader:
        dataset = self._convert_tsv_to_dataset(data_file=dataset_file)
        dataset = self._load_nli_datasets(dataset, threads=threads)
        return DataLoader(dataset, batch_size=batch_size)
