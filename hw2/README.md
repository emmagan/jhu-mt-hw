Important python programs (`-h` for usage):

-`./align` aligns words. It takes an optional `-c` flag for combination type (union vs. intersection).

-`./two-way.py` trains IBM model 1 p(e|f) and p(f|e) and writes the models to e_max.csv and f_max.csv. It take an optional `-i` flag for number of iterations.

-`./align-model-1` and `./ibm-model-1.py` were used to align and train IBM model 1

-`./align-model-original` is the baseline align starter code.

Example usage:

   > python ./two-way.py -n 1000 -i 10
   > python ./align -n 100000 > alignment

The `trained` directory contains trained models as csv files.

-`model-1-1000.csv` is model 1 p(f|e) trained on 50 iterations and 1000 sentences.

-`e_max.csv` is model 1 p(e|f), used for two-way.

-`f_max.csv` is model 1 p(f|e), used for two-way.

The `alignments` directory contains other alignment files.

-`dice.a` is the baseline

-`model-1.a` is model 1 p(e|f)

-`two-way-intersection.a` is two-way intersection alignment file

-`two-way-intersection.a` is two-way union alignment file

The `data` directory contains a fragment of the Canadian Hansards,
aligned by Ulrich Germann:

-`hansards.e` is the English side.

-`hansards.f` is the French side.

-`hansards.a` is the alignment of the first 37 sentences. The 
  notation i-j means the word as position i of the French is 
  aligned to the word at position j of the English. Notation 
  i?j means they are probably aligned. Positions are 0-indexed.