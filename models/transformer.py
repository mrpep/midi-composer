import torch
import pytorch_lightning as pl
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MidiTransformer(pl.LightningModule):
    def __init__(self, model_dim=128, n_classes=356):
        super().__init__()
        self.lut = torch.nn.Embedding(num_embeddings=n_classes, embedding_dim=model_dim)
        self.pos_embedding = PositionalEncoding(model_dim, dropout=0)
        dec_layer = torch.nn.TransformerEncoderLayer(model_dim,8, batch_first=True)
        self.net = torch.nn.TransformerEncoder(dec_layer,6)
        self.classification_layer = torch.nn.Linear(model_dim, n_classes)
        
    def forward(self,x):
        transformer_in = self.pos_embedding(self.lut(x))
        mask = torch.triu(torch.ones(199, 199) * float('-inf'), diagonal=1).to(self.device)
        transformer_out = self.net(transformer_in,mask)
        return self.classification_layer(transformer_out)
    
    def training_step(self,batch, batch_idx):
        logits = self.forward(batch[:,:-1])
        loss = torch.nn.functional.cross_entropy(torch.transpose(logits,1,2),batch[:,1:])
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)