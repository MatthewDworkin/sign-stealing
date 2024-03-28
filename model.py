# Contains the model

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tokenizer import Tokenizer


class signGPT(nn.Module):
    def __init__(self, max_context_len, tokenizer):
        """Initializes the model.
    
        """
        super().__init__()
        self.max_context_len = max_context_len
        self.tokenizer = tokenizer
        self.dense = nn.Linear(tokenizer.total_tokens, tokenizer.vocab_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor):
        """Given a sequence of baseball signs, outputs logits over the classes {none, steal}.

        Args:
            x (torch.Tensor): Of shape (B, max_context_len)

        Returns:
            _type_: _description_
        """
        out = self.dense(x)
        out = self.sigmoid(out)
        return out
    
    def generate(self):
        """Generates a new sequence of baseball signs.

        Returns:
            _type_: _description_
        """
        return NotImplementedError
    

if __name__ == "__main__":
    tokenizer = Tokenizer(32, 23, 2)
    model = signGPT(32, tokenizer)
    fake_x = torch.ones((64, 16))
    logits = model(fake_x)
    model.apply_class_mask()