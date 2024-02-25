import pandas as pd
import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset

from pathlib import Path
import argparse

import time

#script for annotating datasets with guhr pipeline. Inputfiles need to be in feather. Usable via argparse 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                        prog='guhr_senti',
                        description='Add Guhr et al. BERT Sentiment Polarity Classification to data frame.',
                        )

    parser.add_argument('--input', dest="input", type=str, help="Name of the input file. Required. (feather)")
    parser.add_argument('--column', dest="column", default="text", type=str, help="Name of the column containing the "
                                                                                      "input text. If not specified "
                                                                                      "defaults to 'text'")
    parser.add_argument('--output', dest="output", default="output.feather", help="Name of output file, "
                                                                                      "if not specified defaults to "
                                                                                      "'output.feather'")
    args = parser.parse_args()
    p = Path(args.input)
    out = Path(args.output)


    #read dataset
    data = pd.read_feather(p)

    dataset = Dataset.from_pandas(data)


    #defines pipeline and selects guhr model    
    pipe = pipeline("text-classification",model="oliverguhr/german-sentiment-bert",device = "cuda:0")
    
    
    #saves results as feather
    results = []
    start_time = time.process_time()
    for out in pipe(KeyDataset(dataset, "text"), batch_size=32,truncation="only_first"):
        results.append(out['label'])

        
    time_total = time.process_time() - start_time
    with open(f'{p}_time_guhr.txt', 'w') as f:
        f.write(f'time elapssed = {time_total}'  '\n'+f'number of entries = {len(data.index)} ')    
    df = pd.DataFrame(dataset)
    if 'index' in df.columns:
        df.set_index(['index'])
    df['guhr'] = results
    df['guhr'] = df['guhr'].astype(str)
    df = df.reset_index(drop=True)
    df.to_feather(out)