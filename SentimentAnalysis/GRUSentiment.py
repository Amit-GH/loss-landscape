import torch.nn as nn
import torch


class GRUSentiment(nn.Module):
    def __init__(self, vocab=None, input_dim=None, hidden_dim=None, n_layers=None):
        if vocab is None:
            # this model should be loaded from file.
            return

        embedding_dim = 100

        super(GRUSentiment, self).__init__()
        self.embed = nn.Embedding(len(vocab), embedding_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.relu = nn.ReLU(hidden_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        # print("Input: ", x.shape)
        embed_input = self.embed(x)
        embed_input = embed_input.permute(1, 0, 2)
        # print("embed_input: ", embed_input.shape)

        # packed_embedded = nn.utils.rnn.pack_padded_sequence( embed_input, lengths.cpu(), enforce_sorted=False, batch_first=True)
        # print("packed_embedded: ", packed_embedded.data.shape)

        out, h = self.gru(embed_input)
        # (hidden, cell)=h
        # print("H: ", h.shape)

        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(out)
        '''print("output pad packed: ", output.shape)
        print("Output after reshaping: ", output[-1, :, :].shape)'''
        # output=output[-1, :, :]
        # print("Hidden shape: ", hidden.shape)
        # print("Cell shape: ", cell.shape)
        concat_hidden = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        # print("Shape of concatenated hidden layer: ", concat_hidden.shape)
        out = self.fc(concat_hidden)
        # print("Output GRUNET: ", out.shape)
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden
