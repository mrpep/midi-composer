from models import MidiLSTM
from utils import find_last_checkpoint
import random
import torch
from midi_extractors import events_to_midi, messages_to_midi
import numpy as np

model = MidiLSTM()

model.load_from_checkpoint(find_last_checkpoint('checkpoints'))
initial_note = random.randint(64,100)
STEPS_GEN = 1000
BUFFER_LEN = 200
model_in = torch.zeros((1,BUFFER_LEN)).to(dtype=torch.long)
model_in[0,0]=initial_note
generated_sequence = [initial_note]
k = 0

temperature = 0.3
for i in range(STEPS_GEN):
    model_out = model(model_in)
    probs = torch.softmax(model_out/temperature,dim=-1).detach().cpu().numpy()[0,k]

    generated_note = np.random.multinomial(1, probs)
    generated_sequence.append(int(np.argmax(generated_note)))
    k = min(BUFFER_LEN-1,k+1)
    if k >= BUFFER_LEN:
        model_in = torch.tensor([generated_sequence[-BUFFER_LEN:]]).to(dtype=torch.long)
    else:
        model_in[0,k] = generated_sequence[-1]

midiseq = events_to_midi(generated_sequence)
messages_to_midi(midiseq)
