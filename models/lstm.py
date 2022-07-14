import torch
import pytorch_lightning as pl


class MidiLSTM(pl.LightningModule):
    def __init__(self, model_dim=128, n_classes=356):
        super().__init__()
        self.lut = torch.nn.Embedding(num_embeddings=n_classes, embedding_dim=model_dim)
        self.net = torch.nn.Sequential(torch.nn.LSTM(model_dim, model_dim, batch_first=True))
        self.classification_layer = torch.nn.Linear(model_dim, n_classes)
        
    def forward(self,x):
        lstm_in = self.lut(x)
        lstm_out, (hs,cs) = self.net(lstm_in)
        return self.classification_layer(lstm_out)
    
    def training_step(self,batch, batch_idx):
        logits = self.forward(batch[:,:-1])
        loss = torch.nn.functional.cross_entropy(torch.transpose(logits,1,2),batch[:,1:])
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)