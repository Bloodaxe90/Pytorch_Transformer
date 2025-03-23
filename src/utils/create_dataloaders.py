from torch.utils.data import DataLoader
from src.utils.character_dataset import CharacterDataset


def create_dataloaders(
    text: str,
    workers: int,
    train_percent: float = 0.9,
    block_size: int = 5,
    batch_size: int = 32,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:

    train_split: int = int((len(text) - 1) * train_percent)
    train_dataset = CharacterDataset(text[:train_split], block_size)
    test_dataset = CharacterDataset(text[train_split:], block_size)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
    )
    return train_dataloader, test_dataloader
