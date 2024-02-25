# First install
# pip install git+https://git.informatik.uni-leipzig.de/computational-humanities/research/mlmc.git@658edcd433e3a0bd4db6992091550049e4ac0c63
# and hope for the best.

import mlmc
from pathlib import Path
import argparse
import pandas as pd
import time

#script for annotating datasets with mlmc zeroshot. Input-files need to be in feather. Usable via argparse.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='ZeroshotAdding',
                        description='Add Zeroshot Sentiment Polarity Classification to data frame.',
                        )
    parser.add_argument('--input', dest="input", type=str, help="Name of the input file. Required. (feather)")
    parser.add_argument('--column', dest="column", default="text", type=str, help="Name of the column containing the "
                                                                                  "input text. If not specified "
                                                                                  "defaults to 'text'")
    parser.add_argument('--output', dest="output", default="output.feather", help="Name of output file, "
                                                                                  "if not specified defaults to "
                                                                                  "'output.feather'")
    parser.add_argument('--device', dest="device", default="cuda:0", help="Name of the GPU device to use (cuda:XX). If "
                                                                          "not specified defaults to 'cuda:0' ")
    args = parser.parse_args()
    p = Path(args.input)
    out = Path(args.output)



    data = pd.read_feather(p)

    model = mlmc.models.Encoder(classes ={"negative":0,"neutral": 1, "positive": 2},
                                 finetune="bias",
                                 representation="svalabs/gbert-large-zeroshot-nli",
                                 target="single",
                                sformatter=lambda x: f"Die Stimmung in diesem Text ist {x}",
                                         device="cuda:0"
                                        )

    
    start_time = time.process_time()
    data["zeroshot"] = model.predict_batch(data[args.column].to_list(), batch_size=32)
    time_total = time.process_time() - start_time
    print(f'dataset: {p} guhr own library \n time elapssed: {time_total}s for {len(data)} entries. Items per sec = {len(data)/float(time_total)}s ')   
    data["zeroshot"] =  data["zeroshot"].map(lambda x: x[0])
    data.reset_index(drop=True).to_feather(out)
