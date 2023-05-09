import os
from config_files import config
from audio import audio_file_analysis

if __name__ == "__main__":
    """
    This is used to determine whether the textual features should be 
    extracted or the audio features. 
    """
    current_path = os.path.dirname(os.path.realpath(__file__))
    feature_type = config.EXPERIMENT_DETAILS['FEATURE_EXP']
    audio_file_analysis.startup()

    print('Finished Processing')

