import mido
from mido import MidiFile

def midi_to_events(filename, time_resolution=0.01):
    mid = MidiFile(filename)
    messages = []
    messages_int = []
    elapsed_time = 0
    total_time = 0
    #NOTEON - 1:128
    #NOTEOFF - 129:256
    #TIMESHIFTS - 256:356
    for m in mid:
        if (hasattr(m,'velocity') and m.velocity == 0) and (m.type == 'note_on'):
            elapsed_time += m.time
            if elapsed_time > time_resolution:
                messages.append('Time shift: {}'.format(elapsed_time))
                messages_int.append(int(elapsed_time/time_resolution) + 256)
            messages.append('Note_off: {}'.format(m.note))
            messages_int.append(m.note + 128)
            elapsed_time = 0
        elif (hasattr(m,'velocity') and m.velocity > 0) and (m.type == 'note_on'):
            elapsed_time += m.time
            if elapsed_time > time_resolution:
                messages.append('Time shift: {}'.format(elapsed_time))
                messages_int.append(int(elapsed_time/time_resolution) + 256)
            messages.append('Note_on: {}'.format(m.note))
            messages_int.append(m.note)
            elapsed_time = 0
        else:
            elapsed_time += m.time
        if m.time > 0.01:
            total_time += m.time
    return messages_int

def events_to_midi(events, time_resolution=0.01):
    time_shift = 0
    messages = []
    for e in events:
        if e < 128:
            message = mido.Message('note_on',note=e,velocity=127,time=time_shift)
            messages.append(message)
            time_shift = 0
        elif (128<=e) and (e<256):
            message = mido.Message('note_on',note=e-128,velocity=0,time=time_shift)
            messages.append(message)
            time_shift = 0
        else:
            time_shift += (e-256)*time_resolution
    return messages

def messages_to_midi(messages, midi_filename='new_song.mid'):
    mid = MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    for m in messages:
        m.time = int(m.time*300)
        track.append(m)
    mid.save(midi_filename)