from Sentiment import get_train_val_test_data_iterators, train, epoch_time, binary_accuracy, evaluate
from Sentiment import evaluate_extra, evaluate_with_dropout
import time
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext.data as data
from pathlib import Path
from SentimentAnalysis import GRUHaste
from SentimentAnalysis import Explorer
import argparse


def train_model(vocab, input_dim, hidden_dim, dropout, zoneout, criterion,
        train_iter, val_iter):

    model = GRUHaste.GRUNet(vocab, input_dim, hidden_dim, 2, dropout=dropout, zoneout=zoneout)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)

    print("Learning the model.")
    n_epochs = 10
    best_valid_loss = np.inf
    best_model_state_dict = None
    train_loss_list = []
    valid_loss_list = []
    
    # Variabled needed to restore model from an interrupted/failed run
    temporary_model_dict_path = "temporary_values/gru_model.pt"
    temporary_epoch_path = "temporary_values/gru_model_epochs.pt"
    temporary_train_loss_list_path = "temporary_values/gru_model_train_list.pt"
    temporary_valid_loss_list_path = "temporary_values/gru_model_valid_list.pt"
    start_epoch = 0

    # If some epochs were already run, restore the model to that position
    if Path(temporary_model_dict_path).is_file():
        model.load_state_dict(torch.load(temporary_model_dict_path))
        model.eval()
        start_epoch = torch.load(temporary_epoch_path) + 1
        train_loss_list = torch.load(temporary_train_loss_list_path)
        valid_loss_list = torch.load(temporary_valid_loss_list_path)

    for epoch in range(n_epochs - start_epoch):
        epoch = epoch + start_epoch

        start_time = time.time()

        train_loss, train_acc = train(model, train_iter, optimizer, criterion)
        train_loss_list.append(train_loss)
        valid_loss, valid_acc, c_matrix, precision, recall, f1 = evaluate_extra(model, val_iter, criterion)
        valid_loss_list.append(valid_loss)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state_dict = model.state_dict().copy()

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        print("Validation data c_matrix = {}, precision = {:.4g}, recall = {:.4g}, f1 = {:.4g}.".format(
            c_matrix, precision, recall, f1
        ))

        # Save the model so that if code fails, we start from this epoch
        torch.save(model.state_dict(), temporary_model_dict_path)
        torch.save(epoch, temporary_epoch_path)
        torch.save(train_loss_list, temporary_train_loss_list_path)
        torch.save(valid_loss_list, temporary_valid_loss_list_path)
        print("Model saved after epoch {}.".format(epoch + 1))

    print("Best validation loss for dropout {} and zoneout {} is {}".format(
        dropout, zoneout, best_valid_loss
    ))

    # Plot the loss curve
    explorer = Explorer.Explorer()
    explorer.plot_train_test_curves_for(np.arange(start=1, stop=n_epochs+1), train_loss_list, valid_loss_list,
        True, filename_suffix="haste_gru_d{}_z{}".format(dropout, zoneout))
    
    # Delete the saved files as they are no longer needed
    Path(temporary_model_dict_path).unlink()
    Path(temporary_epoch_path).unlink()
    Path(temporary_train_loss_list_path).unlink()
    Path(temporary_valid_loss_list_path).unlink()

    # return the best model
    model.load_state_dict(best_model_state_dict)
    model.eval()
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Parse command line argumnets
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_file', '-mf', required=True, help='file name of the model')
    parser.add_argument('--tune_dropconnect', help='do hyperparameter tuning for dropconnect',
        default=False)
    parser.add_argument('--dropout', help='dropconnect value for training.', default="-1")
    parser.add_argument('--zoneout', help='zoneout value for training.', default="-1")
    parser.add_argument('--hidden_units', help='number of units in each GRU layer.', default="-1")
    args = parser.parse_args()
    
    print("model file passed =", args.model_file)
    print("tune dropconnect =", args.tune_dropconnect)
    ## Parsing completed
    
    saved_model_file_path = "sentiment_models/" + args.model_file

    train_iter, val_iter, test_iter, TEXT = get_train_val_test_data_iterators()
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    # Check if model is already trained
    if Path(saved_model_file_path).is_file():
        model = torch.load(saved_model_file_path)
        print("Model loaded from file {}.".format(saved_model_file_path))
    elif bool(args.tune_dropconnect) == True:
        model = None
        print("Hyperparameter tuning for dropconnect.")
        hidden_dim_list = [64, 128, 256]
        dropout_list = [0.05, 0.1, 0.5]
        input_dim = len(TEXT.vocab)
        vocab = TEXT.vocab
        best_model_state_dict = {}
        best_valid_loss = np.inf
        best_hidden_dim = None
        best_dropout = None
        for hidden_dim in hidden_dim_list:
            for dropout in dropout_list:
                if hidden_dim == 64 or (hidden_dim == 128 and dropout == 0.05):
                    # already calculated
                    continue
                print("\n\n======> Learning for hidden_dim {} and dropout {}.".format(hidden_dim, dropout))
                model = train_model(vocab, input_dim, hidden_dim, dropout, 0, criterion,
                    train_iter, val_iter)
                valid_loss, valid_acc = evaluate(model, val_iter, criterion)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_model_state_dict = model.state_dict().copy()
                    best_hidden_dim = hidden_dim
                    best_dropout = dropout
        model = GRUHaste.GRUNet(vocab, input_dim, best_hidden_dim, 2, dropout=best_dropout, zoneout=0)
        model.load_state_dict(best_model_state_dict)
        model.eval()
        torch.save(model, saved_model_file_path)
        print("Best model with hidden_dim {} and dropout {} has been saved at {}.".format(
            best_hidden_dim, best_dropout, saved_model_file_path
        ))
    else:
        input_dim = len(TEXT.vocab)
        hidden_dim = int(args.hidden_units)
        output_dim = 1
        dropout = float(args.dropout)
        zoneout = float(args.zoneout)

        if hidden_dim == -1 or dropout == -1 or zoneout == -1:
            raise Exception("Invalid value for hidden_units {}, dropout {}, zoneout {}.".format(
                hidden_dim, dropout, zoneout
            ))

        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        vocab = TEXT.vocab
        print("Learning model with fixed params. hidden_dim {} dropout {} zoneout {}.".format(
            hidden_dim, dropout, zoneout
        ))
        model = train_model(vocab, input_dim, hidden_dim, dropout, zoneout, criterion,
            train_iter, val_iter)
        torch.save(model, saved_model_file_path)
        print("Model has been saved at {}.".format(saved_model_file_path))

    print("Checking performance on Test data.")
    # test_loss, test_acc = evaluate(model, test_iter, criterion)
    test_loss, test_acc, test_c_matrix, test_pre, test_rec, test_f1 = evaluate_with_dropout(model, test_iter, criterion)
    # test_loss, test_acc, test_c_matrix, test_pre, test_rec, test_f1 = evaluate_extra(model, test_iter, criterion)
    print("Test data: loss {}, acc {}, precision {}, recall {}, f1 {}.".format(
        test_loss, test_acc, test_pre, test_rec, test_f1
    ))
    # print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
