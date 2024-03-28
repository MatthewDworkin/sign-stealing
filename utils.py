import torch


class TextDataset(torch.utils.data.Dataset):
  def __init__(self, data):
    """Takes tokenized data that is a list of (_, 129) array-likes,
    then concatenates them and makes them into one long (_, 129) tensor of tokens."""
    self.data = torch.cat(data, dim=0)

  def __getitem__(self, index: int):
    self.data[index, :-1], self.data[index, 1:]
    return 

  def __len__(self) -> int:

    return len(self.data)