import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import glob
from progress.bar import IncrementalBar
import warnings
warnings.filterwarnings('ignore')


def get_pages(input_file, output_path, terms):
    input_df = pd.read_csv(input_file)
    frames = []
    print(input_df.columns)
    ner_data = IncrementalBar('ner data for volume', max=len(input_df.index))
    for index, row in input_df.iterrows():
        ner_data.next()
        df = pd.DataFrame(input_df.iloc[index]).transpose()

        for t in terms:
            text = df.loc[df.cleaned_nltk_text.str.contains(t) == True]
            if len(text) > 0:
                
                counts = df.cleaned_nltk_text.apply(lambda x: x.count(t))
                if int(counts) > 0:
                    
                    text['term'] = t
                    text['word_counts'] = counts
                    if os.path.exists(output_path):
                        text.to_csv(output_path, mode='a', header=False, index=False)
                    else:
                        text.to_csv(output_path, header=True, index=False)

    ner_data.finish()
    df = pd.read_csv(output_path)
    df = df.drop_duplicates()
    df.to_csv(output_path, header=True, index=False)


    

if __name__ ==  "__main__" :
    terms = ['congo']
    get_pages('../data/arab_observer_corpus_cleaned.csv', '../data/arab_observer_congo_corpus.csv', terms)
   