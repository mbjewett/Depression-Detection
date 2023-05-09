import os

# Use this string to write a brief detail about the current experiment. This
# string will be saved in a logger for this particular experiment
EXPERIMENT_BRIEF = ''

# Set to complete to use all the data
# Set to sub to use training/dev sets only
# Network options: custom or custom_att (to use the attention mechanism)
# Set to complete to use all the data
# Set to sub to use training/dev sets only
# Network options: custom or custom_att (to use the attention mechanism)
EXPERIMENT_DETAILS = {'FEATURE_EXP': 'mel',
                      'CLASS_WEIGHTS': False,
                      'USE_GENDER_WEIGHTS': False,
                      'SUB_SAMPLE_ND_CLASS': False,  # Make len(dep) == len(
                      # ndep)
                      'CROP': False,
                      'OVERSAMPLE': False,
                      'SPLIT_BY_GENDER': False,  # Only for use in test mode
                      'FEATURE_DIMENSIONS': 120, # this is sequence length. 120 frames is 3.84s 
                      'FREQ_BINS': 40, # this is the feature dimension. 
                      'BATCH_SIZE': 16,
                      'SNV': True,
                      'LEARNING_RATE':1e-3,
                      'SEED': 1000, # keep this constant for reproducibility. 
                      'TOTAL_EPOCHS': 20, #recommmended to run 100 epochs for model convergence.
                      'TOTAL_ITERATIONS': 3280,
                      'ITERATION_EPOCH': 1,
                      'LEARN_RATE_FACTOR': 2,
                      'WEIGHT_DECAY':0,
                      'EXP_RUNTHROUGH': 1}

#EXPERIMENT_DETAILS = {'FEATURE_EXP': 'name_of_custom_feature',
#                       'CLASS_WEIGHTS': False,
#                       'USE_GENDER_WEIGHTS': False,
#                       'SUB_SAMPLE_ND_CLASS': False,  # Make len(dep) == len(
#                       # ndep)
#                       'CROP': False,
#                       'OVERSAMPLE': False,
#                       'SPLIT_BY_GENDER': False,  # Only for use in test mode
#                       'FEATURE_DIMENSIONS': 120, # this is sequence length. 120 frames is 3.84s 
#                       'FREQ_BINS': 12, # this is the feature dimension. 
#                       'BATCH_SIZE': 16,
#                       'SNV': True,
#                       'LEARNING_RATE':1e-2,
#                       'SEED': 1000, # keep this constant for reproducibility. 
#                       'TOTAL_EPOCHS': 2, #recommmended to run 100 epochs for model convergence.
#                       'TOTAL_ITERATIONS': 3280,
#                       'ITERATION_EPOCH': 1,
#                       'LEARN_RATE_FACTOR': 2,
#                       'WEIGHT_DECAY':0,
#                       'EXP_RUNTHROUGH': 1}

EXPERIMENT_BRIEF = ('cnn_lstm'
                                + '_feat_'+EXPERIMENT_DETAILS["FEATURE_EXP"]
                                + '_dim_'+str(EXPERIMENT_DETAILS["FEATURE_DIMENSIONS"])
                                + '_batch_'+str(EXPERIMENT_DETAILS['BATCH_SIZE'])
                                + '_lr_'+str(EXPERIMENT_DETAILS['LEARNING_RATE'])
                                + '_wd_'+str(EXPERIMENT_DETAILS['WEIGHT_DECAY']) 
                                + '_lrf_'+str(EXPERIMENT_DETAILS['LEARN_RATE_FACTOR']))

EXPERIMENT_DETAILS['SUB_DIR'] = ('cnn_lstm'
                                + '_feat_'+EXPERIMENT_DETAILS["FEATURE_EXP"]
                                + '_dim_'+str(EXPERIMENT_DETAILS["FEATURE_DIMENSIONS"])
                                + '_batch_'+str(EXPERIMENT_DETAILS['BATCH_SIZE'])
                                + '_lr_'+str(EXPERIMENT_DETAILS['LEARNING_RATE'])
                                + '_wd_'+str(EXPERIMENT_DETAILS['WEIGHT_DECAY']) 
                                + '_lrf_'+str(EXPERIMENT_DETAILS['LEARN_RATE_FACTOR']))
# Determine the level of crop, min file found in training set or maximum file
# per set (ND / D) or (FND, MND, FD, MD)
MIN_CROP = True
# Determine whether the experiment is run in terms of 'epoch' or 'iteration'
ANALYSIS_MODE = 'epoch'

# How to calculate the weights: 'macro' uses the number of individual
# interviews in the training set (e.g. 31 dep / 76 non-dep), 'micro' uses the
# minimum number of segments of both classes (e.g. min_num_seg_dep=35,
# therefore every interview in depressed class will be normalised according
# to 35), 'both' combines the macro and micro via the product, 'instance'
# uses the total number of segments for each class to determine the weights (
# e.g. there could be 558 dep segs and 440 non-dep segs).
WEIGHT_TYPE = 'instance'

# Set to 'm' or 'f' to split into male or female respectively
# Otherwise set to '-' to keep both genders in the database
GENDER = '-'

# These values should be the same as those used to create the database
# If raw audio is used, you might want to set these to the conv kernel and
# stride values
WINDOW_SIZE = 1024
HOP_SIZE = 512
OVERLAP = int((HOP_SIZE / WINDOW_SIZE) * 100)

FEATURE_FOLDERS = ['audio_data', 'logmel']
EXP_FOLDERS = ['log', 'model', 'condor_logs']

if EXPERIMENT_DETAILS['FEATURE_EXP'] == 'text':
    FEATURE_FOLDERS = None
else:
    FEATURE_FOLDERS = ['audio_data', 'logmel']
EXP_FOLDERS = ['log', 'model', 'condor_logs']

if EXPERIMENT_DETAILS['FEATURE_EXP'] == 'logmel' or EXPERIMENT_DETAILS[
    'FEATURE_EXP'] == 'MFCC' or EXPERIMENT_DETAILS['FEATURE_EXP'] == \
        'MFCC_concat':
    if EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND']:
        FOLDER_NAME = f"BKGND_{EXPERIMENT_DETAILS['FEATURE_EXP']}" \
                      f"_{str(EXPERIMENT_DETAILS['FREQ_BINS'])}"
    elif not EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and \
            EXPERIMENT_DETAILS['REMOVE_BACKGROUND']:
        FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}_{str(EXPERIMENT_DETAILS['FREQ_BINS'])}"
    elif not EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and not \
            EXPERIMENT_DETAILS['REMOVE_BACKGROUND']:
        FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}" \
                      f"_" \
                      f"{str(EXPERIMENT_DETAILS['FREQ_BINS'])}_with_backgnd"
else:
    FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}"

if EXPERIMENT_DETAILS['SNV']:
    FOLDER_NAME = FOLDER_NAME + '_svn_exp'
else:
    FOLDER_NAME = FOLDER_NAME + '_exp'

if EXPERIMENT_DETAILS['USE_GENDER_WEIGHTS']:
    EXPERIMENT_DETAILS['SUB_DIR'] = EXPERIMENT_DETAILS['SUB_DIR'] + '_gen'

PATH_OF_FOLDER='/media/amber/pikachu/vijay/workdir/model_arch/214b_project_update/214b_project/' # path where 214b_project is downloaded
DATASET = PATH_OF_FOLDER+'EATD/EATD-Corpus'
WORKSPACE_MAIN_DIR = PATH_OF_FOLDER+'EATD/audio_feats/feats_DepAudioNet'
WORKSPACE_FILES_DIR = PATH_OF_FOLDER+'eatd_classification_depaudionet'
LABELS_FOLDER=PATH_OF_FOLDER+'EATD/labels/'
TRAIN_SPLIT_PATH =PATH_OF_FOLDER+'EATD/labels/train_EATD_labels.csv'
DEV_SPLIT_PATH =PATH_OF_FOLDER+'EATD/labels/validation_EATD_labels.csv'
FULL_TRAIN_SPLIT_PATH =PATH_OF_FOLDER+'EATD/labels/combined_EATD_labels.csv'
COMP_DATASET_PATH =PATH_OF_FOLDER+'EATD/labels/combined_EATD_labels.csv'
