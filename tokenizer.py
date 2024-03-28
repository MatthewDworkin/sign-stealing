import torch
import numpy as np

class Tokenizer():
  def __init__(self, context_length, vocab_size=23):
    super().__init__()
    self.char_to_token_vocab = {}
    self.token_to_char_vocab = {}
    self.vocab_size = vocab_size
    self.eos = vocab_size
    self.pad = vocab_size + 1
    self.bos = vocab_size + 2
    self.cls0 = vocab_size + 3
    self.cls1 = vocab_size + 4
    self.token_to_char_vocab[self.bos] = "<BOS>" # 25
    self.token_to_char_vocab[self.eos] = "<EOS>" # 23
    self.token_to_char_vocab[self.pad] = "<PAD>" # 24
    self.total_tokens = self.vocab_size + 4

    self.next_int = 0
    self.context_length = context_length

  def tokenize(self, strings):
    """Takes a list[str] of sequences and turns it into an array-like of integer tokens.
    >>> toke = Tokenizer(16)
    >>> strings, labels = ["abc", "cab"], ["0", "1"]
    >>> toke.tokenize(strings, labels)
    tensor([[25.,  0.,  1.,  2., 23., 26., 24., 24., 24., 24., 24., 24., 24., 24.,
             24., 24.],
            [25.,  2.,  0.,  1., 23., 27., 24., 24., 24., 24., 24., 24., 24., 24.,
             24., 24.]])
    """
    tokens = torch.ones((len(strings), self.context_length)) * self.pad

    for i, string in enumerate(strings):
      tokens[i, 0] = self.bos
      for j, char in enumerate(string):
        if char not in self.char_to_token_vocab.keys():
          self.char_to_token_vocab[char] = self.next_int
          self.token_to_char_vocab[self.next_int] = char
          self.next_int += 1
        tokens[i, j+1] = self.char_to_token_vocab[char]
      tokens[i, j+2] = self.eos
    return tokens.to(torch.int)

  def detokenize(self, tokens):
    """Takes an array-like of tokens (integers) and returns the character representation of it."""
    string = ""
    for token in tokens:
      string += self.token_to_char_vocab[token.to(int).item()]
    return string  

if __name__ == "__main__":
   import doctest
   doctest.testmod()