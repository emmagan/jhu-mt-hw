There are four python programs here:

-`decode-original` is the baseline decoder
-`decode` is the beam search swap decoder
-`decode-with-heuristic` is the beam search swap decoder with future cost analysis
-`decode-ext` is the beam search swap decoder with future cost + better reordering model

There is an additional flag (-d) for reordering limits.
Example usage:

    > python decode-ext -k 10 -s 100 -d 5 | python compute-model-score
