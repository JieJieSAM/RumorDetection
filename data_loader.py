import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class RumorDataset(Dataset):
    """
    PyTorch Dataset for health rumor detection.
    Expects lists/arrays of texts and corresponding labels.
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids':      encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels':         torch.tensor(label, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size, shuffle=False, num_workers=0):
    """
    Utility to create a DataLoader from a pandas DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with columns ['text', 'label']
        tokenizer (BertTokenizer): BERT tokenizer instance
        max_len (int): maximum token length per sequence
        batch_size (int): batch size
        shuffle (bool): whether to shuffle data
        num_workers (int): number of worker processes for data loading

    Returns:
        DataLoader
    """
    ds = RumorDataset(
        texts=df['text'].to_numpy(),
        labels=df['label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


if __name__ == '__main__':
    # Example usage
    import config

    # Load dataset from CSV (expects columns: 'text', 'label')
    df_train = pd.read_csv(config.TRAIN_CSV)
    df_val = pd.read_csv(config.VALID_CSV)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)

    # Create DataLoaders
    train_loader = create_data_loader(
        df_train,
        tokenizer,
        max_len=config.MAX_LEN,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    val_loader = create_data_loader(
        df_val,
        tokenizer,
        max_len=config.MAX_LEN,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # Iterate through one batch
    batch = next(iter(train_loader))
    print({k: v.shape for k, v in batch.items()})
