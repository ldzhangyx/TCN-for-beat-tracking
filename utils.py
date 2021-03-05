import os
import librosa
import numpy as np
import yaml
import pdb

# import config
with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


class SingleFile(object):
    def __init__(self, name, path):
        self.name = name
        self.path = path


def init_all_specs(
        input_folder=config['dataset_folder'],
        output_folder=config['spec_folder'],
        n_fft=config['fft_size'],
        hop_length=config['hop_length'],
        n_mels=config['n_mels']):
    """Preprocess spectrograms for all data.

    Args:
        input_folder: raw data path.
        output_folder: spectrogram path
        n_fft: n_fft
        hop_length: hop_length per second
        n_mels: n_mels

    Returns:

    """
    # scan folder and find all *.wav files
    file_list = list()
    for folder in os.listdir(input_folder):

        folder_path = '/'.join([input_folder, folder])
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.wav'):
                    file_path = '/'.join([input_folder, folder, file])
                    file_name = file[:-4]  # cut-off '.wav'
                    file_list.append(SingleFile(file_name, file_path))

                    # create spec for each file
                    spec = init_single_spec(file_path, n_fft, hop_length, n_mels)
                    np.save('/'.join([output_folder, file_name]), spec)
                    print(f'Processed: {file_name}')

    # for index, file in enumerate(file_list):
    #     spec = init_single_spec(file.path, n_fft, hop_length, n_mels)
    #     np.save('/'.join([output_folder, file.name]), spec)
    #     print(f'Processed: {file.name}')


def init_single_spec(
        file_path=None,
        n_fft=config['fft_size'],
        hop_length=config['hop_length'],
        n_mels=config['n_mels']):
    """ Create a spectrogram for single data.

    Args:
        file_path: the file path of single raw data
        n_fft: n_fft
        hop_length: hop length per second
        n_mels: n_mels

    Returns:
        numpy spectrogram

    """
    x, sr = librosa.load(file_path)
    hop_length_in_samples = int(np.floor(hop_length * sr))
    spec = librosa.feature.melspectrogram(
        x,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length_in_samples,
        n_mels=n_mels)
    return np.abs(spec)


# init_all_specs()