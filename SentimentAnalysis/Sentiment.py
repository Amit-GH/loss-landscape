import time

import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext.data as data
from pathlib import Path

from SentimentAnalysis import GRUSentiment


def tokenizer(text):  # create a tokenizer function
    spacy_en = spacy.load('en')
    return [tok.text for tok in spacy_en.tokenizer(text)]


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y)
    acc = sum(correct) / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        text, text_lengths = batch.review
        '''print("Lengths shape: ",text_lengths)
        print("Input shape: ",text.shape)'''

        predictions = model(text, text_lengths)
        predictions = torch.squeeze(predictions)
        '''print("Predictions shape: ", predictions.shape)
        print("True y shape: ", batch.sentiment.shape)'''
        # print(predictions)
        batch.sentiment = batch.sentiment.type_as(predictions)
        loss = criterion(predictions, batch.sentiment)
        # print("Loss: ", loss.item())

        acc = binary_accuracy(predictions, batch.sentiment)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.review

            predictions = model(text, text_lengths)
            predictions = torch.squeeze(predictions)
            batch.sentiment = batch.sentiment.type_as(predictions)
            loss = criterion(predictions, batch.sentiment)

            acc = binary_accuracy(predictions, batch.sentiment)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_train_val_test_data_iterators():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    print("Creating tabular data.")
    start_time = time.time()
    train_data, val_data, test_data = data.TabularDataset.splits(
        path='../dataset/',
        train='train.csv',
        validation='valid.csv',
        test='test.csv',
        format='csv',
        fields=[('review', TEXT), ('sentiment', LABEL)],
        skip_header=True
    )
    print("Time to create tabular data: {} sec.".format(time.time() - start_time))

    print("Building vocabulary from glove embeddings.")
    TEXT.build_vocab(train_data, vectors="glove.6B.100d")
    LABEL.build_vocab(train_data)

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_sizes=(32, 32, 32),
        # the BucketIterator needs to be told what function it should use to group the data.
        sort_key=lambda x: len(x.review),
        sort_within_batch=False,
        device=device
    )
    return train_iter, val_iter, test_iter, TEXT


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iter, val_iter, test_iter, TEXT = get_train_val_test_data_iterators()
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    saved_model_file_path = "../sentiment_models/gru_sentiment.pt"
    # Check if model is already trained
    if Path(saved_model_file_path).is_file():
        model = torch.load(saved_model_file_path)
        print("Model loaded from file {}.".format(saved_model_file_path))
    else:
        input_dim = len(TEXT.vocab)
        hidden_dim = 256
        output_dim = 1
        n_layers = 2
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        model = GRUSentiment.GRUSentiment(TEXT.vocab, input_dim, hidden_dim, n_layers)
        optimizer = optim.Adam(model.parameters())
        model = model.to(device)

        print("Learning the model.")
        n_epochs = 1
        best_valid_loss = np.inf
        for epoch in range(n_epochs):
            start_time = time.time()

            train_loss, train_acc = train(model, train_iter, optimizer, criterion)
            valid_loss, valid_acc = evaluate(model, val_iter, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

        # Save model on disk
        save_model = True
        if save_model:
            torch.save(model, saved_model_file_path)
            print("Model has been saved at {}.".format(saved_model_file_path))

    # Check performance on Test data
    test_loss, test_acc = evaluate(model, test_iter, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
