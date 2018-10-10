# Character-based Word and Context embeddings

This repository contains a usable code from the paper:

_G. Marra, A. Zugarini, S. Melacci, and M. Maggini, “An unsupervised character-aware neural approach to word and context representation learning,” in Proceedings of the 27th International Conference on Artificial Neural Networks – ICANN 2018_

The structure of the project contains:

1. A `data` folder, containing the txt files from which to learn the embeddings.
2. A `log` folder, where the model is saved.
3. The `char2word.py` script, which is the main routine.
4. The `encoder.py` script, which contains some utility functions.


For a standard learning procedure do the following.
* Be sure both `data` and `log` folder are present.
* Put your training data into the `data` folder, with files named as `data(something).txt`, e.g. `data01.txt`.
* Simply run `char2word.py`

The script will create a `vocabulary.txt` file inside the `data` folder to be used during training.
All the configurations are set to the default ones (i.e. the paper ones). The script does not yet provide a command-line configurations (apart for folder configuration, run `--help`for info).
The user willing to have a custom configuration should modify the `Config` configuration class in the `char2word.py` script.

We will provide a more user-friendly command-line interface as soon as possible, together with more details about the training procedure and how to incorporate the model inside bigger models.

To see a fast way to exploit already trained embeddings look at the `SentenceEncoder` class together with the `test` function.
