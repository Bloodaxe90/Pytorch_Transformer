from torch.utils.data import Dataset
import torch


class CharacterDataset(Dataset):

    def __init__(self, input_text, block_size: int = 1):
        super().__init__()
        self.text = input_text
        self.block_size = block_size
        self.characters = sorted(set(input_text))
        self.encode = {char: idx for idx, char in enumerate(self.characters)}
        self.decode = {idx: char for idx, char in enumerate(self.characters)}
        self.tokenised_text = torch.LongTensor(
            [self.encode[char] for char in input_text]
        )

    def __len__(self):
        return len(self.tokenised_text) - (self.block_size + 1)

    def __getitem__(self, idx: int) -> tuple:
        return (
            self.tokenised_text[idx : idx + self.block_size],
            self.tokenised_text[idx + 1 : idx + 1 + self.block_size],
        )
