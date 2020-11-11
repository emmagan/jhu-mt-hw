Our `seq2seq.py` file contains the encoder-decoder.
It has the same arguments as the baseline code.
We added an additional argument, --batch-size, which corresponds to the beam width.

Example usage:

    > python seq2seq.py --batch-size 10 --n_iters 10000 --checkpoint_every 1000

We trained our model on <10000 iterations due to CPU constraints with the default settings.