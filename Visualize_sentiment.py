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

torch.manual_seed(3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_sentiment_model(saved_model_file_path):
    # For google colab these file paths are relative to the python file path.
    # For this project, it has been set to loss-landscape folder. So all paths
    # should be relative to that.
    if Path(saved_model_file_path).is_file():
        model = torch.load(saved_model_file_path)
        return model
    raise Exception("Model not saved in {}.".format(saved_model_file_path))


def extract_model_weights(model):
    return [p.data for p in model.parameters()]


def get_direction_file_path(suffix: str = ""):
    return "sentiment_models/gru_sentiment_direction_file_{}.h5".format(suffix)


def normalize_directions(directions, weights):
    assert len(weights) == len(directions)
    for d, w in zip(directions, weights):
        # TODO: ignore bias weights for normalization. Take care of the last one-node layer.
        d *= w.cpu().norm() / (d.norm() + 1e-8)


def create_random_directions(model):
    weights = extract_model_weights(model)
    directions = [torch.randn(w.size()) for w in weights]
    normalize_directions(directions, weights)
    return directions


def setup_directions(dir_file_path, model):
    if Path(dir_file_path).is_file():
        f = h5py.File(dir_file_path, 'r')
        if "xdirection" in f.keys() and "ydirection" in f.keys():
            f.close()
            print("File {} already exists with directions.".format(dir_file_path))
            return
        f.close()

    # Create the file
    f = h5py.File(dir_file_path, 'w')
    xdirection = create_random_directions(model)
    ydirection = create_random_directions(model)
    h5_util.write_list(f, 'xdirection', xdirection)
    h5_util.write_list(f, 'ydirection', ydirection)
    f.close()
    print("Directions have been set in the file.")


def get_surface_file_path(suffix = ""):
    return "sentiment_models/gru_sentiment_surface_file_{}.h5".format(suffix)


def setup_surface_file(surf_file_path):
    if Path(surf_file_path).is_file():
        f = h5py.File(surf_file_path, 'r')
        if "xcoordinates" in f.keys() and "ycoordinates" in f.keys():
            f.close()
            print("File {} already exists with coordinates.".format(surf_file_path))
            return
        f.close()

    # Create the file
    f = h5py.File(surf_file_path, 'w')
    xcoordinates = np.linspace(-1, 1, 51)
    ycoordinates = np.linspace(-1, 1, 51)
    f['xcoordinates'] = xcoordinates
    f['ycoordinates'] = ycoordinates
    f.close()


def forward_pass_model(model, train_iter):
    criterion = torch.nn.BCEWithLogitsLoss()
    loss, accuracy = evaluate(model, train_iter, criterion)
    return loss, accuracy


def crunch_sentiment(model, weights, directions, train_iter, surf_file_path):
    start_time = time.time()
    loss_key = "train_loss"
    acc_key = "train_acc"
    f = h5py.File(surf_file_path, 'r+')
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:]

    if loss_key not in f.keys():
        shape = (len(xcoordinates), len(ycoordinates))
        losses = -np.ones(shape)
        accuracies = -np.ones(shape)
        f[loss_key] = losses
        f[acc_key] = accuracies
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]

    indices, coords, _ = get_job_indices(losses, xcoordinates, ycoordinates, None)
    for count, index in enumerate(indices):
        if count % 1 == 0:
          print("Evaluating coordinate {}/{}.".format(count, len(indices)))
          print("Time elapsed = {} sec.".format(int(time.time() - start_time)))
        coord = coords[count]
        net_plotter.set_weights(model, weights, directions, coord)
        loss, accuracy = forward_pass_model(model, train_iter)
        losses.ravel()[index] = loss
        accuracies.ravel()[index] = accuracy
        # save this data in file so that it need not be calculated next time
        f[loss_key][:] = losses
        f[acc_key][:] = accuracies
    f[loss_key][:] = losses
    f[acc_key][:] = accuracies
    f.close()
    print("Finished crunch. Total time = {} seconds.".format(int(time.time() - start_time)))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device = ", device)

    model = load_sentiment_model("sentiment_models/gru_sentiment.pt")
    weights = extract_model_weights(model)
    dir_file_path = get_direction_file_path()
    setup_directions(dir_file_path, model)
    surf_file_path = get_surface_file_path()
    setup_surface_file(surf_file_path)
    directions = net_plotter.load_directions(dir_file_path)

    # print("Going to exit.")
    # sys.exit(0)

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
