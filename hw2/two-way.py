import optparse
import sys
import pandas as pd
from collections import defaultdict


optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float",
                     help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int",
                     help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Training with Two way-IBM Model 1...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:opts.num_sents]
t_ef = defaultdict(float)
t_fe = defaultdict(float)
count = defaultdict(float)
total = defaultdict(int)
s_total = defaultdict(float)
# df_ef = pd.DataFrame()
# df_fe = pd.DataFrame()
i = 0

frenchwords = set()
englishwords = set()
for (n, (f, e)) in enumerate(bitext):
    for f_i in f:
        frenchwords.add(f_i)
    for e_i in e:
        englishwords.add(e_i)

# initialize probablity t(e|f) uniformly
for f_i in frenchwords:
    for e_i in englishwords:
        # TODO: should this be set(e) or just e? throughout i am confused :(
        t_ef[(e_i, f_i)] = 1 / (len(frenchwords))
df_ef = pd.Series(t_ef)

# while t hasn't converged, run through an e-m iteration
converged = False
for i in range(10):
    # initialize count and total
    for f_i in frenchwords:
        total[f_i] = 0
        for e_j in englishwords:
            count[(e_j, f_i)] = 0

    for (n, (f, e)) in enumerate(bitext):

        # compute normalization
        for e_j in e:
            s_total[e_j] = 0
            for f_i in f:
                s_total[e_j] += t_ef[(e_j, f_i)]

        # collect counts
        for e_j in e:
            for f_i in f:
                count[(e_j, f_i)] += t_ef[(e_j, f_i)] / s_total[e_j]
                total[f_i] += t_ef[(e_j, f_i)] / s_total[e_j]

    # estimate probabilities
    for f_i in frenchwords:
        for e_j in englishwords:
            t_ef[(e_j, f_i)] = count[(e_j, f_i)] / total[f_i]

    # check convergence with previous iteration
    df_ef = pd.Series(t_ef)
    # converged = df[i].equals(df[i+1])
    # i += 1

count.clear()
total.clear()

for e_i in englishwords:
    for f_i in frenchwords:
        # TODO: should this be set(e) or just e? throughout i am confused :(
        t_fe[(f_i, e_i)] = 1 / (len(englishwords))
df_fe = pd.Series(t_fe)

for i in range(10):
    # initialize count and total
    for e_i in englishwords:
        total[e_i] = 0
        for f_j in frenchwords:
            count[(f_j, e_i)] = 0

    for (n, (f, e)) in enumerate(bitext):

        # compute normalization
        for f_j in f:
            s_total[f_j] = 0
            for e_i in e:
                s_total[f_j] += t_fe[(f_j, e_i)]

        # collect counts
        for f_j in f:
            for e_i in e:
                count[(f_j, e_i)] += t_fe[(f_j, e_i)] / s_total[f_j]
                total[e_i] += t_fe[(f_j, e_i)] / s_total[f_j]

    # estimate probabilities
    for e_i in englishwords:
        for f_j in frenchwords:
            t_fe[(f_j, e_i)] = count[(f_j, e_i)] / total[e_i]

    # check convergence with previous iteration
    df_fe = pd.Series(t_fe)
    # converged = df[i].equals(df[i+1])
    # i += 1

f_max = {}
e_max = {}

with open('f_max.csv', 'w') as file:
    for e in englishwords:
        f_max[e] = df_ef[e].idxmax()
        file.write("%s|%s\n"%(e, f_max[e]))

with open('e_max.csv', 'w') as file:
    for f in frenchwords:
        e_max[f] = df_fe[f].idxmax()
        file.write("%s|%s\n"%(f, e_max[f]))


# intersection = open("two-way-intersection.a", "w")
# union = open("two-way-union.a", "w")
#
# for (f, e) in bitext:
#   for (j, e_j) in enumerate(e):
#     for (k, f_i) in enumerate(f):
#       #print(f_max)
#       if f_i == f_max[e_j] or e_j == e_max[f_i]:
#         union.write(str(k) + "-" + str(j) + " ")
#       if f_i == f_max[e_j] and e_j == e_max[f_i]:
#         intersection.write(str(k) + "-" + str(j) + " ")
#   intersection.write("\n")
#   union.write("\n")
#
# intersection.close()
# union.close()