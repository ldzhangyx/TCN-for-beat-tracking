import torch
from torch.utils.data import Dataset
import numpy as np


import os
import pickle

from madmom.features import DBNBeatTrackingProcessor
import torch

from model import BeatTrackingNet
from utils import init_single_spec
from mir_eval.beat import evaluate
from data import BallroomDataset
from beat_tracker import predict_beats_from_spectrogram

import yaml
import sys
import pdb

# import config
with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def evaluate_model(
        model_checkpoint,
        spectrogram,
        ground_truth):
    """
    Given a model checkpoint, a single spectrogram, and the corresponding
    ground truth, evaluate the model's performance on all beat tracking metrics
    offered by mir_eval.beat.
    """

    prediction = predict_beats_from_spectrogram(
        spectrogram,
        model_checkpoint)

    scores = evaluate(ground_truth, prediction)

    return scores


def evaluate_model_on_dataset(
        model_checkpoint,
        dataset,
        ground_truths):
    """
    Run through a whole instance of torch.utils.data.Dataset and compare the
    model's predictions to the given ground truths.
    """

    # Create dicts to store scores and histories
    mean_scores = {}
    running_scores = {}

    # Iterate over dataset
    for i in range(len(dataset)):
        spectrogram = dataset[i]["spectrogram"].unsqueeze(0)
        ground_truth = ground_truths[i]

        scores = evaluate_model(
            model_checkpoint,
            spectrogram,
            ground_truth)
        beat_scores = scores

        for metric in beat_scores:
            if metric not in running_scores:
                running_scores[metric] = 0.0

            running_scores[metric] += beat_scores[metric]

        # Each iteration, pass our current index and our running score total
        # to a print callback function.
        print(f"{i}, {str(running_scores)}")

    # After all iterations, calculate mean scores.
    for metric in running_scores:
        mean_scores[metric] = running_scores[metric] / (i + 1)

            # Return a dictionary of helpful information
    return {
        "total_examples": i + 1,
        "scores": mean_scores
    }

dataset = BallroomDataset()

ground_truths = (dataset.get_ground_truth(i) for i in range(len(dataset)))

# Run evaluation
evaluate_model_on_dataset(config['default_checkpoint_path'],
    dataset,
    ground_truths)