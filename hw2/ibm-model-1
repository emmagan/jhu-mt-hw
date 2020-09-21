#!/usr/bin/env python
import optparse
import sys
import pandas as pd
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option('-i', '--training-iterations', dest="it", default=5, type="int")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Training with IBM Model 1...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:opts.num_sents]
t = defaultdict(float)
count = defaultdict(float)
total = defaultdict(int)
s_total = defaultdict(float)

frenchwords = set()
englishwords = set()
for (n, (f, e)) in enumerate(bitext):
  for f_i in f:
    frenchwords.add(f_i)
  for e_i in e:
    englishwords.add(e_i)

# initialize probablity t(f|e) uniformly
for f_i in frenchwords:
  for e_i in englishwords:
      t[(f_i,e_i)] = 1/(len(englishwords))
t_prev = pd.Series(t)

# while t hasn't converged, run through an e-m iteration
#converged = False
#while not converged:
  # print("it")
for j in range(opts.it):
  # initialize count and total
  for e_i in englishwords:
    total[e_i] = 0
    for f_j in frenchwords:
      count[(e_i, f_j)] = 0

  for (n, (f, e)) in enumerate(bitext):
    # compute normalization
    for f_i in f:
      s_total[f_i] = 0
      for e_j in e:
        s_total[f_i] += t[(f_i, e_j)]

    # collect counts
    for f_i in f:
      for e_j in e:
        count[(f_i, e_j)] += t[(f_i, e_j)]/s_total[f_i]
        total[e_j] += t[(f_i, e_j)]/s_total[f_i]

  # estimate probabilities
  for e_i in englishwords:
    for f_j in frenchwords:
      t[(f_j, e_i)] = count[(f_j, e_i)]/total[e_i]

  # check convergence with previous iteration
  #converged = t_prev.round(3).equals(pd.Series(t).round(3))
  t_prev = pd.Series(t)

# save trained model in separate file
e_max = {}
for f in frenchwords:
  e_max[f] = t_prev[f].idxmax()
pd.Series(e_max).to_csv('model-1-1000.csv')