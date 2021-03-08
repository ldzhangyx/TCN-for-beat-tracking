import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
import os
import pdb

with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


class BallroomDataset(Dataset):
    """
    A PyTorch Dataset wrapping the ballroom dataset.
    Provides mel spectrograms and a vector of beat annotations per spectrogram frame.
    """

    def __init__(self):
        self.data_list = [item.split('.')[0] for item in os.listdir(config['spec_folder'])]
        self.hop_size = int(np.floor(config['hop_length'] * config['sample_rate']))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_path = config['spec_folder']  + self.data_list[index] + '.npy'
        label_path = config['label_folder'] + self.data_list[index] + '.beats'
        spec = self.get_spectrogram(data_path)
        beat_vector = self.get_beat_vector(label_path, spec)
        spec, beat_vector = self.trim_data(spec, beat_vector)
        spec = torch.from_numpy(np.expand_dims(spec.T, axis=0)).float()
        beat_vector = torch.from_numpy(np.expand_dims(
            beat_vector[:3000].astype('float64'), axis=0)).float()
        return spec, beat_vector

    def get_spectrogram(self, data_path):
        spec = np.load(data_path)
        return spec

    def get_ground_truth(self, index):
        with open(os.path.join(config['label_folder'], self.data_list[index] + '.beats'),'r') as f:
            beat_times = []
            for line in f:
                [beat_time, beat_position]= line.strip().split()
                beat_time = float(beat_time)
                beat_position = int(beat_position)
                if index == 1:
                    beat_times.append(beat_time * config['sample_rate'])
        quantised_times = []
        for time in beat_times:
            spec_frame = int(time / config['hop_length'])
            quantised_time = spec_frame * config['hop_length'] / config['sample_rate']
            quantised_times.append(quantised_time)
        return np.array(quantised_times)

    def get_beat_vector(self, label_path, spec):
        beat_vector = np.zeros(spec.shape[-1])
        beat_list = list()  # parse beat file
        with open(label_path, 'r') as f:
            for line in f:
                [beat_time, beat_position] = line.strip().split()
                beat_list.append((float(beat_time) * config['sample_rate'], int(beat_position)))

        for (beat_time, beat_position) in beat_list:
            spec_frame = min(int(beat_time / self.hop_size), beat_vector.shape[0] - 1)
            for n in range(-2, 3):
                if 0 <= spec_frame + n < beat_vector.shape[0]:
                    beat_vector[spec_frame + n] = 1.0 if n == 0 else 0.5

        return beat_vector

    def trim_data(self, spec, labels):
        x = np.zeros(config['trim_size'])
        y = np.zeros(config['trim_size'][1])

        x_bound = config['trim_size'][0]
        y_bound = min(config['trim_size'][1], spec.shape[1])

        x[:x_bound, :y_bound] = spec[:, :y_bound]
        y[:y_bound] = labels[:y_bound]

        return x, y

# test code
# dataset = BallroomDataset()
# print(dataset[0])