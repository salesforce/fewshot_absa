
# post process raw files (ABSA14)
python prepare_absa14_data.py

# post process raw files (ABSA16)
python prepare_absa16_sb1_data.py
python prepare_absa16_sb2_data.py
python prepare_absa16_sb2_aspect_term.py

# clean train/trial sets
python exclude_train_trial.py

# create LM data (GPT2)
python prepare_lm_data.py
python prepare_lm_data_single.py
python prepare_lm_data_term_category.py

# create SST2 & SST5 data
python prepare_lm_data_sst.py

