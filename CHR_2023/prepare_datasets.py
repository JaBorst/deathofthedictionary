

#Script thats reads raw datasets and prepares the data to be used in quanteda for sentiment analysis. all resulting datasets should be in feather

import pandas as pd
import glob
import os

    

#read folder with scare review data and concernate ist to one feather file. Due to Copyright issues, the SCARE-dataset is not included and needs to be requested from: https://www.romanklinger.de/scare/
def make_feather_scare():
    files = glob.glob("datasets/scare_v1.0.0_data/reviews/*.csv")
    df = []
    for f in files:
        csv = pd.read_csv(f,sep = '\t',usecols=[1,3],names = ['stars','text'],header=None)
        df.append(csv)
    df = pd.concat(df)
    df['text'] = df['text'].astype(str)
    #df = df[df['stars']!= 3]
    df= df.reset_index(drop=True)
    df.to_feather('scare_reviews.feather')    


#read the holidaycheck data from the guhr et al. repository. Due to large file size not included here. can be downloaded at: https://zenodo.org/record/3693810/files/sentiment-data-reviews-and-neutral.zip?download=1
def make_holiday_check():
    holiday_check = pd.read_csv('datasets/holidaycheck/holidaycheck.clean.filtered.tsv',sep = '\t',usecols=[0,1],names = ['stars','text'],header=None)
    # exclude "neutral" reviews. due to different star rating 3 and 4 star ratings are excluded
    holiday_check = holiday_check.query('"stars" ==[1,2,5,6]')
    holiday_check = holiday_check.reset_index(drop=True)
    holiday_check.to_feather('holiday_check.feather')

# read filmstarts data and exclude neutral ratings. save as feather
def filmstarts():
    filmstarts = pd.read_csv('datasets/filmstarts/filmstarts.tsv',sep = '\t',usecols=[1,2],names = ['stars','text'],header=None)
    filmstarts = filmstarts[filmstarts['stars']!=3]
    filmstarts = filmstarts.reset_index(drop=True)
    filmstarts.to_feather('filmstarts.feather')

#read PotTs Tweets, drop mixed sentiment data. Included in the guhr data
def potts():
    val = pd.read_csv("datasets/potts_corpus_label_text.tsv",sep = '\t',usecols=[0,1],names = ['label','text'],header=None)
    val = val[~val['label'].str.contains('mixed')].reset_index(drop=True)
    val['text'] = val['text'].astype(str)
    val.to_feather('Datasets/PotTS.feather')

#read sb10k Tweets, drop mixed sentiment data. Included in the guhr data
def sb10k():
    df = pd.read_csv("datasets/sb10k_corpus_label_text.tsv",sep ='\t',usecols=[0,1],names = ['label','text'],header=None)
    df.to_feather('Datasets/SB10k.feather')


# transform lessing data to feather. Taken from: https://github.com/lauchblatt/LessingSentimentEmotionCorpus
def lessing():    
    lessing = pd.read_csv('datasets/Lessing Sentiment and Emotion Corpus (2018).csv',usecols=[4,6],names=['text','label'],header=None) 
    lessing = lessing[1:].reset_index(drop=True)
    lessing.to_feather('Datasets/lessing_2018.feather')


#read gnd file and transform to feather. can be found at: https://www.informatik.uni-wuerzburg.de/datascience/projects/nlp/kallimachos-concluded/german-novel-dataset/ 
def gnd():
    gnd = pd.read_json('datasets/gnd.json')
    gnd = gnd.rename(columns={'sentence':'text','polarity':'label'})
    gnd.to_feather('gnd.feather')



# SentiLitKrit data from : https://github.com/dkltimon/SentiLitKrit_19-II/tree/master. 
def senti_lit_krit():
    pos = pd.read_table('SentiLitKrit/POS.txt',header=None,usecols=[0],names=['text']) 
    pos['label'] = 'positive'
    neg = pd.read_table('SentiLitKrit/NEG.txt',header=None,usecols=[0],names=['text'])
    neg['label'] = 'negative'
    slk = pd.concat([pos,neg])
    slk = slk.reset_index(drop=True)
    slk.to_feather('slk.feather')
    
    
