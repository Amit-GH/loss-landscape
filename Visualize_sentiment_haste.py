from pathlib import Path

import h5py
import torch
import h5_util
import numpy as np
import time
import net_plotter
import projection
from Sentiment import evaluate, get_train_val_test_data_iterators
from plot_2D import plot_2d_contour
from scheduler import get_job_indices
import sys
import argparse
from Visualize_sentiment import load_sentiment_model, extract_model_weights, \
    normalize_directions, create_random_directions, setup_directions, \
    get_direction_file_path, get_surface_file_path, setup_surface_file, \
    forward_pass_model, crunch_sentiment


torch.manual_seed(3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device = ", device)

    ## Parse command line argumnets
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_file', '-mf', required=True, help='file name of the model')
    parser.add_argument('--visualize_file_suffix','-vfs', required=True, help='suffix to determine the name of direction and surface file')
    args = parser.parse_args()
    
    ## Parsing completed
    print("model file passed =", args.model_file)

    saved_model_file_path = "sentiment_models/" + args.model_file

    model = load_sentiment_model(saved_model_file_path)
    weights = extract_model_weights(model)
    dir_file_path = get_direction_file_path(suffix=args.visualize_file_suffix)
    setup_directions(dir_file_path, model)
    surf_file_path = get_surface_file_path(suffix=args.visualize_file_suffix)
    setup_surface_file(surf_file_path)
    directions = net_plotter.load_directions(dir_file_path)

    # find cosine similarity between x and y directions
    similarity = projection.cal_angle(
        projection.nplist_to_tensor(directions[0]),
        projection.nplist_to_tensor(directions[1])
    )
    print("Cosine similarity between X and Y axis is {0:.4g}.".format(similarity))

    train_iter, val_iter, test_iter, TEXT = get_train_val_test_data_iterators()
    crunch_sentiment(model, weights, directions, train_iter=train_iter, surf_file_path=surf_file_path)

    print("Print the plot")
    plot_2d_contour(surf_file_path, "train_loss")

    print("End of main.")
