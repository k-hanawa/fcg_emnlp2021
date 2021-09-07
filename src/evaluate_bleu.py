# -*- coding: utf-8 -*-

from nltk import bleu_score
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i')

    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding='utf-16', sep='\t', index_col=0)
    sf = bleu_score.SmoothingFunction().method3

    refs = [str(s) for s in df['gold'].tolist()]
    hyps = [str(s) for s in df['output 0'].tolist()]

    bleu = bleu_score.corpus_bleu([[ref.split(' ')] if ref != '[None]' else ['None'] for ref in refs], [hyp.split(' ') for hyp in hyps], smoothing_function=sf)
    print(bleu)

if __name__ == "__main__":
    main()
