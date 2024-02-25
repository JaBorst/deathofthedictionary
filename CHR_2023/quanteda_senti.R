# load qaunteda
library(quanteda)
library(readtext)

library("quanteda.sentiment")

library(arrow)

library(readxl)

library(readr)



# set directory where dictionary files and datasets are located
dicts_senti <- function(path_wd,input,output,text_column= 'text'){
  
  
  #sdirectory first where input dataset and bpw+gpc files are
  setwd(path_wd)
  
  #reads data as df
  senti_df <- arrow::read_feather(input)
  
  #makes quanteda corpus-object from df
  corp <- corpus(senti_df,text_field = text_column)
  
  #Tokenize corpus and pre-process (remove punctuations, numbers)
  toks <- tokens(corp, remove_punct = TRUE, remove_numbers = TRUE)
  


  ### quanteda.sentiment aporach with sentiws dict and binary polarity values calculated via quanteda.sentiment https://rdrr.io/github/quanteda/quanteda.sentiment/src/R/textstat_polarity.R ###
  ##  calculates polarity-score via log of (positive / negative) counts
  start_time <- Sys.time()
  # compute sentiment via quanteda.sentiment on tokenized sentences 
  pol_sentiws <- textstat_valence(toks,dictionary = data_dictionary_sentiws)
  end_time_senti <- Sys.time() - start_time
  print(paste0('time elapsed senti ws : ',end_time_senti))
  #add polarity-values to senti_df-df and if stop-words affect polarity-values adds them as own columns 
  senti_df <- cbind(senti_df, sentiws = pol_sentiws$sentiment)
  
  
  #BPW Dictionary Construction ###
  
  # Reads Excel-file and extracts each polaritylist as a list
  bpw_pos <- data.frame(c(read_excel("BPW_Dictionary.xlsx",sheet=2,col_types = "text",col_names = "word")))
  
  bpw_neg <- data.frame(c(read_excel("BPW_Dictionary.xlsx",sheet=1,col_types = "text",col_names = "word")))
  
  # creates a quanteda dictionary from the wordlists
  bpw_dict <- quanteda::dictionary(list(
    positive = c(bpw_pos[['word']]),
    negative = c(bpw_neg[['word']])))
  
  # defines polarity for the bpw_dict
  polarity(bpw_dict) <- list(pos = "positive", neg = "negative")

  
  # calculates polarity-values for df and binds results as extra column 
  start_time <- Sys.time()
  pol_bpw <- textstat_polarity(toks,dictionary = bpw_dict)
  end_time_BPW <- Sys.time() - start_time
  print(paste0('time elapsed BPW : ',end_time_BPW))
  
  senti_df <- cbind(senti_df, bpw = pol_bpw$sentiment)
  
  
  
  # Dictionary German Polarity Clues by Waltinger 2010 http://www.ulliwaltinger.de/sentiment/ in the 2012 version
  
  pos_gpc <- read_tsv('GermanPolarityClues-2012/GermanPolarityClues-Positive-21042012.tsv',col_names = FALSE,show_col_types = FALSE)
  pos_gpc <- pos_gpc['X1']
  
  
  neg_gpc <- read_tsv('GermanPolarityClues-2012/GermanPolarityClues-Negative-21042012.tsv',col_names = FALSE,show_col_types = FALSE)
  neg_gpc <- neg_gpc['X1']
  
  
  neut_gpc <- read_tsv('GermanPolarityClues-2012/GermanPolarityClues-Neutral-21042012.tsv',col_names = FALSE,show_col_types = FALSE)
  neut_gpc <- neut_gpc['X1']
  # first 19 entries are non-alphanumeric symbols and must be removed for quanteda
  neut_gpc <- tail(neut_gpc,-19)
  
  # creates a quanteda dictionary from the wordlists
  germ_pol_clues <- quanteda::dictionary(list(
    positive = c(pos_gpc['X1']),
    negative = c(neg_gpc['X1']),
    neutral = c(neut_gpc['X1']))
    )
  # defines polarity for the german polarity clues dictionary
  polarity(germ_pol_clues) <- list(pos = "positive", neg = "negative", neut = "neutral")

  
  
  start_time <- Sys.time()
  # calculates polarity-values for df and binds results as extra column
  pol_gpc <- textstat_polarity(toks,dictionary = germ_pol_clues)
  
  end_time_gpc <- Sys.time() - start_time
  print(paste0('time elapsed GPC : ',end_time_gpc))
  
  senti_df <- cbind(senti_df, gpc = pol_gpc$sentiment)
  
  
  
  ### BAWL-R Dictionary Approach. Data taken from: https://www.ewi-psy.fu-berlin.de/psychologie/arbeitsbereiche/allgpsy/Download/index.html
  
  bawl_r <- read_excel('BAWL-R.xlsx')
  bawl_r <- bawl_r[, which((names(bawl_r) %in% c('WORD_LOWER','EMO_MEAN'))==TRUE)]
  
  bawl_r_dict <- quanteda::dictionary(list(emo = c(bawl_r$WORD_LOWER)))
  
  valence(bawl_r_dict) <- list(emo = c(bawl_r$EMO_MEAN))
  
  
  start_time <- Sys.time()
  pol_bawl_r <-  textstat_valence(toks,dictionary = bawl_r_dict)
  
  end_time_bawl <- Sys.time() - start_time
  print(paste0('time elapsed BAWL-R : ',end_time_bawl))
  
  senti_df <- cbind(senti_df, bawl_r = pol_bawl_r$sentiment)
  

  
  #SentiLitKrit Dictionary ###
  
  # Reads Excel-file and extracts each polaritylist as a list
  slk_pos <- read.table('SentiLitKrit/Positive_2018-01-28.txt',sep='\n',header = FALSE,encoding="latin1")
  
  slk_neg <-  read.table('SentiLitKrit/Negative_2018-01-28.txt',sep='\n',header = FALSE,encoding="latin1")
  
  # creates a quanteda dictionary from the wordlists
  slk_dict <- quanteda::dictionary(list(
    positive = c(slk_pos),
    negative = c(slk_neg)))
  
  # defines polarity for the slk_dict
  polarity(slk_dict) <- list(pos = "positive", neg = "negative")
  
  
  # calculates polarity-values for df and binds results as extra column 
  start_time <- Sys.time()
  pol_slk <- textstat_polarity(toks,dictionary = slk_dict)
  end_time_slk <- Sys.time() - start_time
  print(paste0('time elapsed SentiLitKrit : ',end_time_slk))
  
  senti_df <- cbind(senti_df, slk = pol_slk$sentiment) 
  
  print(paste0('number of rows: ',nrow(senti_df)))
  
  #write DF as feather/ nan values are causing trouble, when saving as feather so are set to empty string
  
  senti_df[is.na(senti_df)] <- " "
  arrow::write_feather(senti_df,output)
  
}






