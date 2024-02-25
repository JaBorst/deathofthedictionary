##Script to create classification reports for all tested methods and datsets. use eval dicts or eval_dicts_bin if binary dataset

import pandas as pd

from sklearn.metrics import classification_report


#prepares sentiment scores to absolute values
def prep_dfs_4_eval(gs):
    
    gs.loc[gs['sentiws']<0,'sentiws'] = 'negative'
    gs.loc[gs['sentiws']>0,'sentiws'] = 'positive'
    gs.loc[gs['sentiws']==0,'sentiws'] = 'neutral'
    
    gs.loc[gs['gpc']<0,'gpc'] = 'negative'
    gs.loc[gs['gpc']>0,'gpc'] = 'positive'
    gs.loc[gs['gpc']==0,'gpc'] = 'neutral'
    
    gs.loc[gs['bpw']<0,'bpw'] = 'negative'
    gs.loc[gs['bpw']>0,'bpw'] = 'positive'
    gs.loc[gs['bpw']==0,'bpw'] = 'neutral'
    
    gs.loc[gs['bawl_r']<0,'bawl_r'] = 'negative'
    gs.loc[gs['bawl_r']>0,'bawl_r'] = 'positive'
    gs.loc[gs['bawl_r']==0,'bawl_r'] = 'neutral'
    
    
    gs.loc[gs['slk']<0,'slk'] = 'negative'
    gs.loc[gs['slk']>0,'slk'] = 'positive'
    gs.loc[gs['slk']==0,'slk'] = 'neutral'
    



    
# for sentiment evaluation with more than 2 label, input is a dataset in feather, output defines only name of evaluation csv     
def eval_dicts(input_file,output_name,goldstandard):

    gs = pd.read_feather(input_file)

    #adjust germeval_gs_senti scores
    gs = prep_dfs_4_eval(gs)
 


    sentiws = classification_report(gs[goldstandard], gs['sentiws'],output_dict=True)
    sentiws =pd.DataFrame(sentiws)
    sentiws['method'] = 'sentiws_dict'
    
    
    slk = classification_report(gs[goldstandard], gs['slk'],output_dict=True)
    slk= pd.DataFrame(slk)
    slk['method'] = 'slk'
    
    bpw = classification_report(gs[goldstandard], gs['bpw'],output_dict=True)
    bpw =pd.DataFrame(bpw)
    bpw['method'] = 'bpw_dict'
    
    bawl_r = classification_report(gs[goldstandard], gs['bawl_r'],output_dict=True)
    bawl_r =pd.DataFrame(bawl_r)
    bawl_r['method'] = 'bawl_r'
    
    gpc = classification_report(gs[goldstandard], gs['gpc'],output_dict=True)
    gpc =pd.DataFrame(gpc)
    gpc['method'] = 'german_polarity_clues'
   
    zs = classification_report(gs[goldstandard], gs['zeroshot'],output_dict=True)
    zs =pd.DataFrame(zs)
    zs['method'] = 'zeroshot'
    

    
    guhr_pipe = classification_report(gs[goldstandard], gs['guhr'],output_dict=True)
    guhr_pipe =pd.DataFrame(guhr_pipe)
    guhr_pipe['method'] = 'guhr_pipe'
    
     
    metrics_df = pd.concat([bpw,slk,bawl_r,sentiws,gpc,zs,guhr_pipe])

        
    metrics_df = metrics_df.set_index(['method',metrics_df.index])
    metrics_df.to_csv(f'metrics_val/metrics_{output_name}.csv')
    #metrics_df.to_latex(f"table_{output_name}.tex")    


def eval_dicts_bin(input_file,output_name,goldstandard):

    gs = pd.read_feather(input_file)
    

    gs = prep_dfs_4_eval(gs)
    #adjust germeval_gs_senti scores to values 1,0,-1


    sentiws = classification_report(gs[goldstandard], gs['sentiws'],labels=[-1,1],output_dict=True)
    sentiws =pd.DataFrame(sentiws)
    sentiws['method'] = 'sentiws_dict'
    
    
    slk = classification_report(gs[goldstandard], gs['slk'],labels=[-1,1],output_dict=True)
    slk= pd.DataFrame(slk)
    slk['method'] = 'slk'
    
    bpw = classification_report(gs[goldstandard], gs['bpw'],labels=[-1,1],output_dict=True)
    bpw =pd.DataFrame(bpw)
    bpw['method'] = 'bpw_dict'
    
    bawl_r = classification_report(gs[goldstandard], gs['bawl_r'],labels=[-1,1],output_dict=True)
    bawl_r =pd.DataFrame(bawl_r)
    bawl_r['method'] = 'bawl_r'
    
    gpc = classification_report(gs[goldstandard], gs['gpc'],labels=[-1,1],output_dict=True)
    gpc =pd.DataFrame(gpc)
    gpc['method'] = 'german_polarity_clues'
   
    zs = classification_report(gs[goldstandard], gs['zeroshot'],labels=[-1,1],output_dict=True)
    zs =pd.DataFrame(zs)
    zs['method'] = 'zeroshot'
    

    
    guhr_pipe = classification_report(gs[goldstandard], gs['guhr'],labels=[-1,1],output_dict=True)
    guhr_pipe =pd.DataFrame(guhr_pipe)
    guhr_pipe['method'] = 'guhr_pipe'
    
            
    metrics_df = pd.concat([bpw,slk,bawl_r,sentiws,gpc,zs,guhr_pipe])

        
    metrics_df = metrics_df.set_index(['method',metrics_df.index])
    metrics_df.to_csv(f'metrics/metrics_{output_name}.csv')
    #metrics_df.to_latex(f"table_{output_name}.tex")  


