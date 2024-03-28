

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model import *
from tokenizer import Tokenizer
from tqdm import tqdm, trange

device = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_SEQ_LEN = 32

def load_data(filepath):
    """
    Loads the sequence data.

    """
    with open(filepath, 'r') as file:
        # Read the entire content of the file into a string
        content = file.read()
        file.close()
    data_by_line = content.splitlines()
    data_and_labels = [d.split(", ") for d in data_by_line]
    strings = [d[0] for d in data_and_labels]
    labels = np.array([d[1] for d in data_and_labels], dtype=np.int32)
    toke = Tokenizer(MAX_SEQ_LEN)
    tokens = toke.tokenize(strings)
    return tokens, labels, toke

def main():
    """Runs the training loop.

    Returns:
        int: 0 if successful.

    """

    print(f"Using device {device}")

    # Load in the data
    batch_size = 64
    num_epochs = 1
    # data = load_data("./data_example.txt")
    data, labels, tokenizer = load_data("./train.txt")
    print(data)
    # Set up the model
    signGPT_model = signGPT(tokenizer=tokenizer, max_context_len=MAX_SEQ_LEN).to(device)

    # Begin training
    print(f"Beginning training")
    


    return 0

if __name__ == "__main__":

    main()