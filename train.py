from re import I
from pytorch_lightning.loggers import WandbLogger
from datasets import MaestroDataset, load_maestro
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from models import MidiLSTM, MidiTransformer
from pathlib import Path
from utils import find_last_checkpoint

maestro_metadata = load_maestro('../datasets/maestro-v3.0.0')

ds = MaestroDataset(maestro_metadata)
dl = DataLoader(ds,batch_size=512,shuffle=True)

model = MidiLSTM()

wandb_logger = WandbLogger(project="MIDILSTM")
callbacks = [pl.callbacks.ModelCheckpoint('checkpoints', save_top_k=-1)]
#last_ckpt = None
last_ckpt = find_last_checkpoint('checkpoints')

trainer = pl.Trainer(accelerator='gpu',devices=[1], logger=wandb_logger, callbacks=callbacks, resume_from_checkpoint=last_ckpt)
trainer.fit(model,dl)