import midi
import copy
import numpy as np

import tqdm, mido


def mergeTrack(s):
    """
    Merge all tracks in s in a single one.
    """
    singletrack = midi.Track()
    events = []
    for i, track in enumerate(s):
        t = 0
        for event in track:
            t += event.tick
            if event.name in ['Note On', 'Note Off']:
                candidate = {'t': t, 'event': event}
                if candidate not in events:
                    events.append(candidate)


    events = sorted(events, key=lambda k: (k['t'], k['event'].data[0]))
    tick = 0
    for e in events:
        e['event'].tick = e['t'] - tick
        tick = e['t']
        singletrack.append(e['event'])
    return singletrack

def parseMIDI(midi_file_path):
    """
    Process the MIDI in midi_file_path to extract
    the sequence of notes (dT, T, P). Timing is in 
    MIDI tick representation. The function also 
    returns the tick per beat (tpb) metaparameter
    """ 
    s = midi.read_midifile(midi_file_path)
    tpb = float(s.resolution)
    events = mergeTrack(s)
    T = []
    P = []
    dT = []
    dt = 0
    for n, event in enumerate(events):
        if event.name == 'Note On' and event.data[1] > 0:
            pitch_n = event.data[0]
            n2 = n
            duration_n = 0
            while True:
                n2 += 1
                if n2 > (len(events)-1):
                    break
                duration_n += events[n2].tick
                if events[n2].data[0] == pitch_n and events[n2].name == 'Note Off':
                    break
                if events[n2].data[0] == pitch_n and events[n2].name == 'Note On' and events[n2].data[1] == 0:
                    break
            if duration_n > 0.:
                P.append(pitch_n)
                T.append(duration_n)
                dT.append(event.tick+dt)
            dt = 0
        elif event.name == 'Note Off' or event.data[1] == 0:
            dt += event.tick
    
    #Tick (integer) to beat fraction (float)
    dT = [float(dt)/tpb for dt in dT]
    T = [float(t)/tpb for t in T]
    return dT, T, P, tpb

def getDictionaries(dataset, durations=None):
    p_text = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'bB', 'B']

    dictionaries = {"T": [], "P": [], "dT": [], "P_text": [], "dT_text": [], "T_text": []}
    for key in ['dT', 'T', 'P']:
        flatten = []
        for label, score in dataset.items():
            for x in score[key]:
                flatten.append(x)

        dictionaries[key] = sorted(list(set(flatten)))
    for p in dictionaries['P']:
        dictionaries['P_text'].append(p_text[p%12]+str(p//12))
    if durations is None:
        for dt in dictionaries['dT']:
            dictionaries['dT_text'].append(str(dt))
        for t in dictionaries['T']:
            dictionaries['T_text'].append(str(t))
    else:
        for dt in dictionaries['dT']:
            dictionaries['dT_text'].append(durations['text'][durations['val'].index(dt)])
        for t in dictionaries['T']:
            dictionaries['T_text'].append(durations['text'][durations['val'].index(t)])
        
    return dictionaries


def findClosest(val, vec):
    """
    find the element closest to val in vec 
    """
    diff = [abs(float(el)-float(val)) for el in vec]
    idx = np.argmin(diff)
    return vec[idx]

def tokenize(dataset, dictionaries):
    xP = []
    xT = []
    xdT = []
    labels = []
    for label, melody in dataset.items():
        xP.append([dictionaries["P"].index(x) for x in melody["P"]])
        xT.append([dictionaries["T"].index(x) for x in melody["T"]])
        xdT.append([dictionaries["dT"].index(x) for x in melody["dT"]])
        labels.append(label)
    return xdT, xT, xP, labels

def writeMIDI(dtseq, Tseq, pitchseq, path, label="1", tag="1", resolution=256, bpm=100):

    # Instantiate a MIDI Pattern (contains a list of tracks)
    pattern = midi.Pattern(format = 0, resolution = resolution)
    # Instantiate a MIDI Track (contains a list of MIDI events)
    track = midi.Track()
    # Append the track to the pattern
    pattern.append(track)
    tick = 0
    Events = []

    #Set the tempo
    e = midi.SetTempoEvent()
    e.set_bpm(bpm)
    track.append(e)
    
    #Timing
    timeperbeat = 60. / bpm#[s]
    timepertick = timeperbeat/resolution
    longenough = False
    for dt, T, p in zip(dtseq, Tseq, pitchseq):
        if tick >= resolution*8:#more than 8 beats
            longenough = True
        if dt == 'START/END' or T == 'START/END' or p == 'START/END':
            break
        tick = tick + int(dt*resolution)
        Events.append({'t': tick, 'p': p, 'm': 'ON'})
        Events.append({'t': tick+int(T*resolution), 'p': p, 'm': 'OFF'})

    Events = sorted(Events, key=lambda k: k['t'])
    tick = 0
    for event in Events:
        if event['m'] == 'ON':
            e =  midi.NoteOnEvent(tick=event['t']-tick, velocity=90, pitch=event['p'])
        if event['m'] == 'OFF':
            e =  midi.NoteOffEvent(tick=event['t']-tick, velocity=90, pitch=event['p'])
        track.append(e)
        tick = event['t']
        

    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    if longenough:
        # Save the pattern to disk
        midi.write_midifile(path+label+"_"+tag+".mid", pattern)
    return pattern

def longMIDI(dtseqs, Tseqs, pitchseqs, path, label="1", tag="all", resolution=1024, write_all=False, bpm=80):
    ended = False
    # Instantiate a MIDI Pattern (contains a list of tracks)
    pattern = midi.Pattern(format = 0, resolution = resolution)
    # Instantiate a MIDI Track (contains a list of MIDI events)
    track = midi.Track()
    # Append the track to the pattern
    pattern.append(track)
    tick = 0
    Events = []

    #Set the tempo
    e = midi.SetTempoEvent()
    e.set_bpm(bpm)
    track.append(e)

    i = 0
    number = 0
    for dtseq, Tseq, pitchseq in zip(dtseqs, Tseqs, pitchseqs):
        i+=1
        last_dur = 0
        song_ended = False
        for dt, T, p in zip(dtseq, Tseq, pitchseq):
            if dt == 'START/END' or T == 'START/END' or p == 'START/END':
                song_ended = True
                tick = tick + int(resolution*last_dur) + resolution*4
                break
            tick = tick + int(dt*resolution)
            Events.append({'t': tick, 'p': p, 'm': 'ON'})
            Events.append({'t': tick+int(T*resolution), 'p': p, 'm': 'OFF'})
            last_dur = T
        tick = tick + int(resolution*last_dur) + resolution*4
        if song_ended:
            writeMIDI(dtseq, Tseq, pitchseq, path=path, label=label, tag=str(i), resolution=resolution, bpm=bpm)
            number += 1
        else:
            writeMIDI(dtseq, Tseq, pitchseq, path=path, label=label, tag=str(i)+"_NoEnd", resolution=resolution, bpm=bpm)
    Events = sorted(Events, key=lambda k: k['t'])
    tick = 0
    onehour = False
    for event in Events:
        if event['m'] == 'ON':
            e =  midi.NoteOnEvent(tick=event['t']-tick, velocity=90, pitch=event['p'])
        if event['m'] == 'OFF':
            e =  midi.NoteOffEvent(tick=event['t']-tick, velocity=90, pitch=event['p'])
        track.append(e)
        tick = event['t']
        if tick * (60. / (120.*resolution)) > 3600.:
            onehour = True
            break
      
    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    # Save the pattern to disk
    if write_all:
        midi.write_midifile(path+label+"_"+tag+".mid", pattern)
        midifile = mido.MidiFile(path+label+"_"+tag+".mid")
    
        print(midifile.length, tick * (60. / (120.*resolution)), len(midifile.tracks), midifile.type)
        if onehour:
            print("Long enough...YEY")
    return number

def sample(preds, temperature=1.):
    if temperature == 0.:
        return np.argmax(preds)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    
    return np.argmax(probas)
    out = np.zeros(shape=preds.shape, dtype='float32')
    out[song_idx,t,np.argmax(probas)] = 1.
    return out

def sampleNmax(preds, N=2):
    return sample(preds)
    candidateidxes = np.argsort(preds)[-N:]
    prcandidates = preds[candidateidxes]
    return candidateidxes[sample(prcandidates)]


