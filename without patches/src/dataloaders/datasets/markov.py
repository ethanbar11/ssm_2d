from typing import List
import numpy as np
import torch

def generate_abc_dataset(
        num_sequences: int,
        seq_length: int,
) -> List[str]:
    """
    Create a simple dataset using a Markov chain on 3 characters.
    :param num_sequences: Number of sequences to generate.
    :param seq_length: Length of each generated sequence.
    :return: list containing generated sequences
    """
    # Create a Markov chain on 3 characters
    # chars = list('012')
    chars = list(range(3))
    prior_probs = [0.6, 0.2, 0.2]
    transition_probs = {
        0: [0.3, 0.2, 0.5],
        1: [0.4, 0.5, 0.1],
        2: [0.05, 0.0, 0.95],
    }

    masks = []
    dataset = []
    for i in range(num_sequences):

        # Keep track of ith generated sequence
        mask = []
        seq_i = []

        # Sample from prior
        char = np.random.choice(chars, p=prior_probs)
        mask.append(1)
        seq_i.append(char)
        for j in range(seq_length - 1):
            # Sample from transition distribution
            char = np.random.choice(chars, p=transition_probs[seq_i[-1]])
            #seq_i.append(" ")
            # seq_i[-1] = seq_i[-1]
            mask.append(1)
            seq_i.append(char)

        masks.append(mask)
        dataset.append(seq_i)
        #dataset.append(torch.LongTensor([seq_i]))

    # return (dataset, masks)
    return dataset

def torch_markov_data(n_seq, l_seq):
    data = generate_abc_dataset(n_seq, l_seq)
    return torch.LongTensor(data)


if __name__ == '__main__':
    print(generate_abc_dataset(num_sequences=3, seq_length=4))
    print(torch_markov_dataset(3, 4))
