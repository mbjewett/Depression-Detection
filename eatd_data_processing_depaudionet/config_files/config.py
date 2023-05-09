import os
import numpy as np

# FEATURE_EXP: logmel, mel, raw, MFCC, MFCC_concat, or text
# WHOLE_TRAIN: This setting is for mitigating the variable length of the data
# by zero padding
# SNV will normalise every file to mean=0 and standard deviation=1
EXPERIMENT_DETAILS = {'FEATURE_EXP': 'mel',
                      'FREQ_BINS': 40,
                      'DATASET_IS_BACKGROUND': False,
                      'WHOLE_TRAIN': False,
                      'WINDOW_SIZE': 1024,
                      'OVERLAP': 50,
                      'SNV': True,
                      'SAMPLE_RATE': 16000,
                      'REMOVE_BACKGROUND': True}
#EXPERIMENT_DETAILS = {'FEATURE_EXP': 'name_of_custom_feature',
#                       'FREQ_BINS': 12, # number of features
#                       'DATASET_IS_BACKGROUND': False,
#                       'WHOLE_TRAIN': False,
#                       'WINDOW_SIZE': 1024,
#                       'OVERLAP': 50,
#                       'SNV': True,
#                       'SAMPLE_RATE': 16000,
#                       'REMOVE_BACKGROUND': True}

# Set True to split data into genders
GENDER = False
WINDOW_FUNC = np.hanning(EXPERIMENT_DETAILS['WINDOW_SIZE'])
FMIN = 0
FMAX = EXPERIMENT_DETAILS['SAMPLE_RATE'] / 2
HOP_SIZE = EXPERIMENT_DETAILS['WINDOW_SIZE'] -\
           round(EXPERIMENT_DETAILS['WINDOW_SIZE'] * (EXPERIMENT_DETAILS['OVERLAP'] / 100))

if EXPERIMENT_DETAILS['FEATURE_EXP'] == 'text':
    FEATURE_FOLDERS = None
else:
    FEATURE_FOLDERS = ['audio_data', EXPERIMENT_DETAILS['FEATURE_EXP']]

PATH_OF_FOLDER='/media/amber/pikachu/vijay/workdir/model_arch/214b_project_update/214b_project/' # path where 214b_project is downloaded
DATASET =PATH_OF_FOLDER+'EATD/'
WORKSPACE_MAIN_DIR =PATH_OF_FOLDER+'EATD/audio_feats/feats_DepAudioNet'
WORKSPACE_FILES_DIR =PATH_OF_FOLDER+'eatd_data_processing_depaudionet/'
ALL_SPLIT_PATH = PATH_OF_FOLDER+'EATD/labels/combined_EATD_labels.csv'
TRAIN_SPLIT_PATH =PATH_OF_FOLDER+'EATD/labels/train_EATD_labels.csv'
DEV_SPLIT_PATH =PATH_OF_FOLDER+'EATD/labels/validation_EATD_labels.csv'
FULL_TRAIN_SPLIT_PATH =PATH_OF_FOLDER+'EATD/labels/combined_EATD_labels.csv'
COMP_DATASET_PATH =PATH_OF_FOLDER+'EATD/labels/combined_EATD_labels.csv'
ACOUSTIC_FEATURE=PATH_OF_FOLDER+'custom_features/name_of_custom_feature' # Path where other acoustic feature are stored. 
