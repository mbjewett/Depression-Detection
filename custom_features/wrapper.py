# Authors: Vijay Ravi, Jinhan Wang, Abeer Alwan
# Date: 04/25/2023

import pandas as pd
import numpy as np
import csv
import librosa
import os
#import torchaudio
import opensmile

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
    audio, fs = librosa.load(wav_path, sr=None)  

    # openSMILE features
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals,
                           )
    features=smile.process_signal(audio,fs)
    feat_out_openSMILE = features.to_numpy()

    # librosa features
    #audio,fs = torchaudio.load(wav_path)
    #audio = audio.np().reshape(-1)
    audio.reshape(-1)
    mels = librosa.feature.melspectrogram(y=audio, sr=fs)
    zcr = librosa.feature.zero_crossing_rate(audio)
    lpc = librosa.lpc(y=audio, order=12)
    mfccs = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=13)
    feat_out_1 = np.append(np.array([np.nanmean(mels)]), np.array([np.nanmean(zcr)]))
    feat_out_2 = np.append(np.array([lpc]),np.array(np.nanmean(mfccs, axis=1)))
    feat_out_3 = np.append(np.array([np.nanmedian(mels)]), np.array([np.nanmedian(zcr)]))
    feat_out_4 = np.array(np.nanmedian(mfccs, axis=1))
    feat_out_librosa_mean = np.append(feat_out_1,feat_out_2)
    feat_out_librosa_median = np.append(feat_out_3,feat_out_4)
    feat_out_librosa = np.append(feat_out_librosa_mean,feat_out_librosa_median)

    # append both openSMILE and librosa
    feat_out = np.append(np.array(feat_out_openSMILE)[0,:],feat_out_librosa)
    # to eliminate potential nan values
    feat_out = np.nan_to_num(feat_out)
    # to eliminate potential infinity values
    feat_out = np.nan_to_num(feat_out, posinf=0)

    # output feature index:
    #       0-87:    openSMILE
    #       88:      melspectrogram/mean
    #       89:      zero_crossing_rate/mean
    #       90-102:  lpc
    #       103-115: mfccs/mean
    #       116:     melspectrogram/median
    #       117:     zero_crossing_rate/median
    #       118-130: mfccs/median
    print('Processed {}, dimension {}'.format(wav_path, feat_out.shape))
    return feat_out
    

if __name__ == "__main__":

    PATH_OF_FOLDER='/home/michi/214b_project/' # path where 214b_project is downloaded
    feat_scp_path = PATH_OF_FOLDER+ 'EATD/labels/feat.scp'
    custom_feat_folder=PATH_OF_FOLDER+'custom_features/new_feats' # Path where other acoustic feature should be stored. 

    if not os.path.exists(custom_feat_folder):
        os.mkdir(custom_feat_folder)

    feat_dict = read_feat_scp(feat_scp_path, custom_feat_folder)

    # parse through the feat dict and extract feature for each wav file and store it in the 
    # corresponding path

    for  wav_path,csv_path in feat_dict.items():
        feature = extract_features(PATH_OF_FOLDER +  wav_path)

        # assuming feature is a np array. 
        pd.DataFrame(feature).to_csv(csv_path, header=None, index=None)


