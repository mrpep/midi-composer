from pathlib import Path

def find_last_checkpoint(dir):
    ckpt_files = list(Path(dir).rglob('*.ckpt'))
    epoch_number = [int(f.stem.split('=')[1].split('-')[0]) for f in ckpt_files]
    last_ckpt = ckpt_files[epoch_number.index(max(epoch_number))]
    return last_ckpt