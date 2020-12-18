import torch.nn as nn
import haste_pytorch as haste
import torch

emb_dim=100

class GRUNet(nn.Module):
    def __init__(self, vocab, input_dim, hidden_dim, n_layers, dropout, zoneout):
        super(GRUNet, self).__init__()
        self.embed = nn.Embedding(len(vocab), emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)   
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        #self.relu=nn.ReLU(hidden_dim)
        self.tanh=nn.Tanh()
        self.gru1 = haste.GRU(input_size=emb_dim, hidden_size=hidden_dim, dropout=dropout,zoneout=zoneout, batch_first=True)
        self.gru2 = haste.GRU(input_size=hidden_dim, hidden_size=hidden_dim, dropout=dropout,zoneout=zoneout, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, 1)
        
        
    def forward(self, x, lengths):
        embed_input = self.embed(x)
        embed_input=embed_input.permute(1,0,2)#torch.Size([32, 522, 100])
        out, h1 = self.gru1(embed_input)
        h1=h1.permute(1,0,2)
        out, h2 = self.gru2(self.tanh(h1)) #torch.Size([32, 1264, 256])
        out = self.fc(torch.cat((h2.squeeze(), h1.squeeze()), dim=1))
        return out
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden
