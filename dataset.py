import datasets
import re
import tqdm
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class TranslationDataset(Dataset):
    def __init__(self, dataset, de_tokenizer = None, en_tokenizer = None, max_length=64):
        """
        args:
            dataset (datasets.Dataset)
        """
        self.dataset = dataset
        self.de_tokenizer = de_tokenizer
        self.en_tokenizer = en_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        element = self.dataset[idx]
        return {'de': list(self.de_tokenizer(element['de'], max_length=self.max_length, padding='max_length').values()), 
                'en': list(self.en_tokenizer(element['en'], max_length=self.max_length, padding='max_length').values())}

def translation_collate(list_of_dicts):
    batch = {'de': [], 'en': []}
    
    for key in ['de', 'en']:
        for dictionary in list_of_dicts:
            batch[key].append(dictionary[key])
        
        batch[key] = torch.tensor(batch[key]).transpose(0,1)

    return batch

def load_clean_dataset(de_tokenizer, en_tokenizer, max_length):
    ds = load_from_disk('data/wmt17_de-en_cleaned.hf')

    train_ds = TranslationDataset(ds['train'], de_tokenizer, en_tokenizer, max_length)
    val_ds = TranslationDataset(ds['validation'], de_tokenizer, en_tokenizer, max_length)
    test_ds = TranslationDataset(ds['test'], de_tokenizer, en_tokenizer, max_length)

    return train_ds, val_ds, test_ds

def clean_dataset(dataset, min_length=5, max_length=64, max_ratio=1.5):
    dataset = dataset.copy()
    whitelist = set(
        "abcdefghijklmnopqrstuvwxyz ÄÖÜäöüßABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥")

    def clean_text(text):
        # Remove non-UTF8 characters
        text = text.encode("utf-8", "ignore").decode("utf-8")

        # Remove URLs and HTML tags
        text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", text)
        text = re.sub(r"<.*?>", "", text)

        # Remove characters not in the whitelist
        text = ''.join(c for c in text if c in whitelist)

        return text

    cleaned_dataset = datasets.DatasetDict()

    for split in dataset.keys():
        data_split = {
            'de': [],
            'en': []
        }

        for data in tqdm.tqdm(dataset[split], desc = split):
            src_text = data["translation"]["de"]
            tgt_text = data["translation"]["en"]

            # Clean source and target texts
            src_text = clean_text(src_text)
            tgt_text = clean_text(tgt_text)

            # Check if the lengths are within the specified range
            if min_length <= len(src_text) <= max_length and min_length <= len(tgt_text) <= max_length:
                # Check the ratio between source and target lengths
                ratio = len(src_text) / len(tgt_text)
                if 1/max_ratio <= ratio <= max_ratio:
                    data_split['en'].append(src_text)
                    data_split['de'].append(tgt_text)
        cleaned_dataset[split] = datasets.Dataset.from_dict(data_split)
    return cleaned_dataset


if __name__ == '__main__':
    ds = load_dataset("wmt17", "de-en")
    cleaned_dataset = clean_dataset(ds)
    cleaned_dataset.save_to_disk("data/wmt17_de-en_cleaned.hf")


