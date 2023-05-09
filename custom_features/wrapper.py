# Authors: Vijay Ravi, Jinhan Wang, Abeer Alwan
# Date: 04/25/2023

import pandas as pd
import numpy as np
import csv
import librosa
import os

'''
    This python wrapper file can be used for extraction of custom features and 
    saving it into csv files. Please modify the file paths and feature extraction function before 
    running this. 
'''

def read_feat_scp(feat_scp_path, custom_feat_folder):
    '''
        This function creates a dictionary where the key value is the audio path and 
        value is the path where the feature must be stored. 
        Eg: {'EATD/EATD-Corpus/t_104/positive_out.wav':	'211104.csv'}

        Inputs are 
            1. feat_scp_path: path where feats.scp given to you is saved.
            2. custom_feat_folder: folder where the current feature being extracted are to be saved. 
        
        Output is
            1. a dictionary consisting of key value pairs as shown in the example above.
    '''

    reader = csv.reader(open(feat_scp_path, 'r'), delimiter = '\t')
    feat_dict = {}
    for wav_path,csv_path in reader:
        feat_dict[wav_path] = custom_feat_folder+'/'+csv_path
    
    return feat_dict

def extract_features(wav_path):

    '''
    This function is for extracting you custom feature. 
    The input is the path of the audio file.
    The ouput should be the matrix of features extractd 
    For the feature matrix, 
        - feature dimension is number of rows
        - frame dimension is number of columns. 
        - for ex. Mel features will have 80xnumber of frames as matrix shape. 
    '''
    audio_data, sample_rate = librosa.load(wav_path, sr=None)
    feature = <call function to extract features>(audio_data) 
    print('Processed {}, dimension {}'.format(wav_path, feature.shape))
    
    return feature # try to format features as np array with shape as mentioned above. 


if __name__ == "__main__":

    PATH_OF_FOLDER='/home/jinhan/workdir/214b_project/' # path where 214b_project is downloaded
    feat_scp_path = PATH_OF_FOLDER+ 'EATD/labels/feat.scp'
    custom_feat_folder=PATH_OF_FOLDER+'custom_features/name_of_custom_Feature' # Path where other acoustic feature should be stored. 

    if not os.path.exists(custom_feat_folder):
        os.mkdir(custom_feat_folder)

    feat_dict = read_feat_scp(feat_scp_path, custom_feat_folder)

    # parse through the feat dict and extract feature for each wav file and store it in the 
    # corresponding path

    for  wav_path,csv_path in feat_dict.items():
        feature = extract_features(PATH_OF_FOLDER +  wav_path)

        # assuming feature is a np array. 
        pd.DataFrame(feature).to_csv(csv_path, header=None, index=None)


