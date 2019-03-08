import spacy
import pandas as pd
import os, sys
import glob
import nltk
# nltk.download('punkt')
import string
from nltk.corpus import stopwords
from nltk.util import ngrams
# nltk.download('stopwords')
from progress.bar import IncrementalBar
import itertools
import warnings
warnings.filterwarnings('ignore')
nlp = spacy.load('en_core_web_lg')
# print(nlp.Defaults.stop_words)

def custom_tokenize(text):
    if not text:
#       print('The text to be tokenized is a None type. Defaulting to blank string.')
        text = ''
    return nltk.word_tokenize(text)

def process_text(df, stopping):
    '''Clean the text using both nltk and spacy. Nltk is more permissive, whereas we use spacy to clean out all numbers and OCR error. So cleaned_nltk_texts is lightly cleaned data, and cleaned_spacy_text is heavily filtered data.'''
    df_1 = df[0:0]
    processing = IncrementalBar('processing text', max=len(df.index))
    for index, row in df.iterrows():
        processing.next()
        raw_text = row['google_vision_text']
        row['original_text'] = raw_text
        row['cleaned_nltk_text'] = ''
        row['cleaned_spacy_text'] = ''
        row['original_counts'] = len(row.google_vision_text.split())
        row['cleaned_nltk_counts'] = 0
        row['cleaned_spacy_counts'] = 0
        tokens = custom_tokenize(raw_text)
        page_terms = ''
        for t in tokens:
            # if stopping:
            if t.lower() in string.punctuation:
                continue
            elif t.lower() in stopwords.words('english'):
                continue
            else:
                page_terms += t.lower() + ' '
            # else:
                    # page_terms += t.lower() + ' '
        row.cleaned_nltk_text = page_terms
        spacy_terms = ''
        spacy_text = nlp(page_terms)
        for ent in spacy_text:
            if len(ent.ent_type_) > 0 or ent.is_alpha:
                if( ent.is_punct == False) and (any(i.isdigit() for i in ent.text) == False) and (ent.is_stop ==False):
                    if '.' in ent.text:
                        text = ('').join(ent.text.split('.'))
                        spacy_terms += text + ' '
                    else:
                        spacy_terms += ent.text + ' '
        row.cleaned_spacy_text = spacy_terms
        df_1 = df_1.append(row, ignore_index=True)  

    df_1.reset_index(inplace=True)
    processing.finish()
    df_1.cleaned_nltk_counts = df_1.cleaned_nltk_text.str.split().str.len()
    df_1.cleaned_spacy_counts = df_1.cleaned_spacy_text.str.split().str.len()
    return df_1


def get_imagelucida_csv(stopping, aggregating, output_path):
    '''Function to combine years of data collected through Image Lucida. The data is processed using nltk and spacy'''
    df_1960 = pd.read_csv('../data/arab_observer_1960_imagelucida.csv')
    df_1961 = pd.read_csv('../data/arab_observer_1961_imagelucida2.csv')
    df_1962 = pd.read_csv('../data/arab_observer_1962_imagelucida.csv')
    df_1960.google_vision_text.fillna('', inplace=True)
    df_1961.google_vision_text.fillna('', inplace=True)
    df_1962.google_vision_text.fillna('', inplace=True)
    df = pd.concat([df_1960, df_1961, df_1962])

    df['vol'] = df.file_name.str.split('/').str[-1]
    df.vol = df.vol.str.split('_').str[:-2].str.join('_')
    
    data = process_text(df, stopping)
    if aggregating: 
        df_grouped_spacy = data.groupby(['vol'])['spacy_text'].apply(' '.join).reset_index()
        
        df_grouped_spacy_counts = data.groupby(['vol'])['spacy_counts'].sum().reset_index()
        df_grouped_tokenized = data.groupby(['vol'])['tokenized_text'].apply(' '.join).reset_index()
        df_grouped_tokenized_counts = data.groupby(['vol'])['tokenized_counts'].sum().reset_index()
        df_grouped_pages = data.groupby(['vol'])['page_number'].count().reset_index()
        final_df = pd.merge(df_grouped_spacy, df_grouped_tokenized, on=['vol'])
        df_1 = pd.merge(final_df, df_grouped_spacy_counts, on=['vol'])
        df_2 = pd.merge(df_1, df_grouped_tokenized_counts, on=['vol']) 
        data = pd.merge(df_2, df_grouped_pages, on=['vol']) 
    def update_vals(row):
        if 'no' in row.vol:
            term = row.vol.split('_')[0:-1]
            year = row.vol.split('_')[-4]
            term.append(year)
            row.vol = ('_').join(term)
            return row
        else:
            return row
    df = data
    df = df.apply(update_vals, axis=1)
    df['year'] = df.vol.str.split('_').str[-1]

    df['day'] = df.vol.str.split('_').str[-2]

    df['month'] = df.vol.str.split('_').str[-3]
    df['date'] = df.year + '-' + df.month + '-' + df.day
    df.to_csv(output_path)

if __name__ ==  "__main__" :
    print(len(sys.argv))
    stopping = False
    aggregating = False
    if len(sys.argv) > 1:
        stopping = sys.argv[1]
        aggregating = sys.argv[2]
    get_imagelucida_csv(True, False, '../data/arab_observer_1960_1962_wo_stopwords.csv')
