import utils
import numpy as np
import pickle
from tqdm import tqdm
import os, sys
from sys import stdout

import keras
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model, load_model
from keras.layers import Input, Masking, TimeDistributed, Dense, Concatenate, Dropout, LSTM, GRU
from keras.optimizers import Adam

class BachProp:
    """
    Class defining BachProp
    """

    def __init__(self, corpus, TBPTT_size=128, batch_size=32):
        self.datapath = "../data/"
        self.corpus = corpus

        self.loadmodelpath = "../load/"
        if not os.path.exists("../save/"):
            os.makedirs("../save/")
        self.outpath = "../save/"+corpus+'/'
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
        self.genpath = self.outpath + "midi/"
        if not os.path.exists(self.genpath):
            os.makedirs(self.genpath)
        

        self.TBPTT_size = TBPTT_size
        self.batch_size = batch_size


        self.IO = None
        self.dataset = None
        
        self.log = {'loss': [], 'val_loss': [], 
                        'dT_acc': [], 'val_dT_acc': [], 
                        'T_acc': [], 'val_T_acc': [], 
                        'P_acc': [], 'val_P_acc': [], 
                        'dT_loss': [], 'val_dT_loss': [],
                        'T_loss': [], 'val_T_loss': [],
                        'P_loss': [], 'val_P_loss': []}

        self.best_val_accP = 0.
        self.best_tr_accP = 0.
        self.best_val_accT = 0.
        self.best_tr_accT = 0.
        self.best_val_accdT = 0.
        self.best_tr_accdT = 0.

        self.split = None
        self.maxlen = None


    def processData(self):
        dataset = {}
        print("Reading MIDI files")
        for filename in tqdm(os.listdir(self.datapath+self.corpus+"/midi/")):
            if filename[-3:] in ["mid", "MID", "SQU", "KPP", "squ", "kpp"]:
                label = filename[:-4]
                try:
                    dTseq, Tseq, Pseq, tpb = utils.parseMIDI(self.datapath+self.corpus+"/midi/"+filename)

                except TypeError:
                    print(label, 'skipped')
                    continue
                dataset[label] = {}
                dataset[label]['T']= Tseq
                dataset[label]['dT']= dTseq
                dataset[label]['P']= Pseq
                dataset[label]['TPB'] = tpb
                dPseq = []
                for n, p in enumerate(Pseq):
                    if n == 0:
                        dPseq.append(0)
                    else:
                        dPseq.append(p-Pseq[n-1])
                dataset[label]['dP'] = dPseq

        #Construct all possible durations
        bases = [1./16, 1./12]#64 and triplet of 32
        increments = []
        for b in bases:
            m = 0
            while m*b < 1.:
                increments.append(m*b)
                m+=1
        allowed_durations = []
        for d in range(64): #For all durations up to 64 quarter notes
            for i in increments:
                allowed_durations.append(round(d+i, 5))
        increments = sorted(list(set(increments)))
        allowed_durations = sorted(list(set(allowed_durations)))

        print("Rhythm Normalization")
        #Convert to the set of possible durations
        for label, score in tqdm(dataset.items()):
            score['dT'] = [utils.findClosest(dt, allowed_durations) for dt in score['dT']]
            score['T'] = [utils.findClosest(t, allowed_durations) for t in score['T']]
            

        dictionaries = {}
        for key in ['dT', 'T', 'P']:
            flatten = []
            for label, score in dataset.items():
                for x in score[key]:
                    flatten.append(x)
            if key == 'P':
                dictionaries[key] = list(range(min(flatten), max(flatten)+1))
            else:
                dictionaries[key] = sorted(list(set(flatten)))

        self.dataset = dataset
        self.dictionaries = dictionaries

        print(self.dictionaries)

    def saveData(self):
        pickle.dump([self.dataset, self.dictionaries], open(self.datapath+self.corpus+"/data.pkl", "wb" ))

    def loadData(self):
        if os.path.exists(self.datapath+self.corpus+"/data.pkl"):
            print("Loading Data")
            self.dataset, self.dictionaries = pickle.load( open(self.datapath+self.corpus+"/data.pkl", "rb" ) )
        else:
            print("No preprocessed data for %s has been found -> Processing %s"%(self.corpus, self.datapath+self.corpus))
            self.processData()
            self.saveData()
        #Add start/end tags
        tag='START/END'
        if tag not in self.dictionaries['dT']:
            self.dictionaries['dT'].append(tag)
            self.dictionaries['T'].append(tag)
            self.dictionaries['P'].append(tag)

    def computeIntervalRepr(self):
        
        for label in self.dataset.keys():
            dPseq = []
            p_nm1 = self.dataset[label]['P'][0]
            for n, p_n in enumerate(self.dataset[label]['P']):
                dPseq.append(p_n-p_nm1)
                p_nm1 = p_n
            self.dataset[label]['dP'] = dPseq

    def checkRepresentation(self):
        label = str(np.random.choice(list(self.dataset.keys()), 1)[0])
        utils.writeMIDI(self.dataset[label]['dT'][1:-1], self.dataset[label]['T'][1:-1], self.dataset[label]['P'][1:-1], path="../data/"+self.corpus+"/", label=label, tag='retrieved')



    def ANN2data(self, XdTs, XTs, XPs):
        """
        Translate back from one-hot matrices to note sequences
        """
        dTs = []
        Ts = []
        Ps = []

        tag='START/END'
        if tag not in self.dictionaries['dT']:
            self.dictionaries['dT'].append(tag)
            self.dictionaries['T'].append(tag)
            self.dictionaries['P'].append(tag)

        for XdT, XT, XP in zip(XdTs[:,1:], XTs[:,1:], XPs[:,1:]):
            xdT = np.where(XdT == 1)[1]
            xT = np.where(XT == 1)[1]
            xP = np.where(XP == 1)[1]
            dT = []
            T = []
            P = []
            for dt, t, p in zip(xdT, xT, xP):
                dt = self.dictionaries['dT'][dt]
                t = self.dictionaries['T'][t]
                p = self.dictionaries['P'][p]

                dT.append(dt)
                T.append(t)
                P.append(p)

            dTs.append(dT)
            Ts.append(T)
            Ps.append(P)
            
        return dTs, Ts, Ps

    def getMaxLen(self):
        seqlens = [len(score['P']) for score in self.dataset.values()]
        len95Perc = int(np.mean(seqlens)+2*np.std(seqlens))
        maxlen = len95Perc + int(self.TBPTT_size-len95Perc%(self.TBPTT_size))
        self.maxlen = maxlen
        return maxlen

    def data2ANN(self):

        if self.dataset is None:
            self.loadData()


        #Add start/end tags
        tag='START/END'
        if tag not in self.dictionaries['dT']:
            self.dictionaries['dT'].append(tag)
            self.dictionaries['T'].append(tag)
            self.dictionaries['P'].append(tag)
        for label, score in self.dataset.items():
            if tag not in score['dT']:
                score['dT'].insert(0,tag)
                score['T'].insert(0,tag)
                score['P'].insert(0,tag)
                score['dT'].append(tag)
                score['T'].append(tag)
                score['P'].append(tag)


        xdT, xT, xP, labels = utils.tokenize(self.dataset, self.dictionaries)

        P = [np_utils.to_categorical(x, len(self.dictionaries['P'])) for x in xP]
        T = [np_utils.to_categorical(x, len(self.dictionaries['T'])) for x in xT]
        dT = [np_utils.to_categorical(x, len(self.dictionaries['dT'])) for x in xdT]

        seqlens = [len(X) for X in P]
        len95Perc = int(np.mean(seqlens)+2*np.std(seqlens))

        maxlen = len95Perc + int(self.TBPTT_size-len95Perc%(self.TBPTT_size)) + 1

        dT = pad_sequences(dT, value=0., dtype="int32", padding="post", truncating="post", maxlen=maxlen)
        T = pad_sequences(T, value=0., dtype="int32", padding="post", truncating="post", maxlen=maxlen)
        P = pad_sequences(P, value=0., dtype="int32", padding="post", truncating="post", maxlen=maxlen)
        #print("Longest melody length [note]: %i, mean: %i, std: %i, selected length: %i"%(max(seqlens),np.mean(seqlens),np.std(seqlens), maxlen))

        XdT = np.asarray([x[:-1] for x in dT], dtype=int)
        YdT = np.asarray([x[1:] for x in dT], dtype=int)

        XT = np.asarray([x[:-1] for x in T], dtype=int)
        YT = np.asarray([x[1:] for x in T], dtype=int)

        XP = np.asarray([x[:-1] for x in P], dtype=int)
        YP = np.asarray([x[1:] for x in P], dtype=int)

        print("\tFinal I/O data shape:")
        print('\tdT', XdT.shape)
        print('\tT', XT.shape)
        print('\tP', XP.shape)


        self.TBPTT_steps = []
        for X in XP:
            steps = 0
            while np.sum(X[steps*self.TBPTT_size:(steps+1)*self.TBPTT_size]) > 0:
                steps += 1
            self.TBPTT_steps.append(steps)
        self.TBPTT_steps = np.asarray(self.TBPTT_steps)

        IO = {'XP': XP, 'YP': YP, 'XT': XT, 'YT': YT, 'XdT': XdT, 'YdT': YdT, 'TBPTT_steps': self.TBPTT_steps, "labels": labels}

        self.IO = IO
        self.maxlen = maxlen - 1


    def buildModel(self, dropout=0., recurrent_dropout=0.):
        print('\nBuilding Model')

        X = dict()
        M = dict()
        H = dict()
        Y = dict()
        
        dTvocsize = len(self.dictionaries['dT'])
        Tvocsize = len(self.dictionaries['T'])
        Pvocsize = len(self.dictionaries['P'])
        
        X['dT_n'] = Input(batch_shape=(self.batch_size, self.TBPTT_size, dTvocsize), name='XdT_n')
        X['dT_np1'] = Input(batch_shape=(self.batch_size, self.TBPTT_size, dTvocsize), name='XdT_np1')
        X['T_n'] = Input(batch_shape=(self.batch_size, self.TBPTT_size, Tvocsize), name='XT_n') 
        X['T_np1'] = Input(batch_shape=(self.batch_size, self.TBPTT_size, Tvocsize), name='XT_np1') 
        X['P_n'] = Input(batch_shape=(self.batch_size, self.TBPTT_size, Pvocsize), name='XP_n')

        M['dT_n'] = Masking(mask_value=0.)(X['dT_n'])
        M['dT_np1'] = Masking(mask_value=0.)(X['dT_np1'])
        M['T_n'] = Masking(mask_value=0.)(X['T_n'])
        M['T_np1'] = Masking(mask_value=0.)(X['T_np1'])
        M['P_n'] = Masking(mask_value=0.)(X['P_n'])

        X['H1'] = Concatenate()([M['dT_n'], M['T_n'], M['P_n']])
        H['1'] =  GRU(128, 
            return_sequences=True, 
            stateful=True
            )(X['H1'])

        X['H2'] = Concatenate()([M['dT_n'], M['T_n'], M['P_n'], H['1']])
        H['2'] =  GRU(128, 
            return_sequences=True, 
            stateful=True
            )(X['H2'])

        X['H3'] = Concatenate()([M['dT_n'], M['T_n'], M['P_n'], H['2']])
        H['3'] =  GRU(128, 
            return_sequences=True, 
            stateful=True
            )(X['H3'])

        X['H4'] = H['3']
        H['4'] =  GRU(128, 
            return_sequences=True, 
            stateful=True
            )(X['H4'])
   
        X['HP'] = Concatenate()([H['1'], H['2'], H['3'], H['4'], M['dT_np1'], M['T_np1']])
        H['P'] = TimeDistributed(Dense(Pvocsize, activation='relu'))(X['HP'])
        Y['P'] = TimeDistributed(Dense(Pvocsize, activation='softmax'), name='YP')(H['P'])

        self.Pmodel = Model(inputs = [X['dT_n'], X['T_n'], X['P_n'], X['dT_np1'], X['T_np1']], outputs = [Y['P']])
        self.Pmodel.compile(
            loss='categorical_crossentropy', 
            optimizer=Adam(),
            metrics=['acc'],
            loss_weights=[.6])
    
        # H['1'].trainable = False
        # H['2'].trainable = False
        # H['3'].trainable = False

        X['HdT'] = H['1']
        H['dT'] = TimeDistributed(Dense(dTvocsize, activation='relu'))(X['HdT'])
        Y['dT'] = TimeDistributed(Dense(dTvocsize, activation='softmax'), name='YdT')(H['dT'])
        self.dTmodel = Model(inputs = [X['dT_n'], X['T_n'], X['P_n']], outputs = [Y['dT']])
        self.dTmodel.compile(
            loss='categorical_crossentropy', 
            optimizer=Adam(),
            metrics=['acc'],
            loss_weights=[.1])

        X['HT'] = Concatenate()([H['1'], H['2'], M['dT_np1']])
        H['T'] = TimeDistributed(Dense(Tvocsize, activation='relu'))(X['HT'])
        Y['T'] = TimeDistributed(Dense(Tvocsize, activation='softmax'), name='YT')(H['T'])
        self.Tmodel = Model(inputs = [X['dT_n'], X['T_n'], X['P_n'], X['dT_np1']], outputs = [Y['T']])
        self.Tmodel.compile(
            loss='categorical_crossentropy', 
            optimizer=Adam(),
            metrics=['acc'],
            loss_weights=[.3])


        

        #H['2'].trainable = False
        #H['T'].trainable = False

        
    def loadModel(self, tag=None):
        print('\nLoading Model')
        if tag is None:
            self.dTmodel = load_model(self.loadmodelpath+self.corpus+"/dTmodel.bp")
            self.Tmodel = load_model(self.loadmodelpath+self.corpus+"/Tmodel.bp")
            self.Pmodel = load_model(self.loadmodelpath+self.corpus+"/Pmodel.bp")
        else:
            self.dTmodel = load_model(self.loadmodelpath+self.corpus+"/"+tag+"_dTmodel.bp")
            self.Tmodel = load_model(self.loadmodelpath+self.corpus+"/"+tag+"_Tmodel.bp")
            self.Pmodel = load_model(self.loadmodelpath+self.corpus+"/"+tag+"_Pmodel.bp")

        self.log = pickle.load(open(self.loadmodelpath+self.corpus+"/log.bp", "rb" ))
        self.best_val_accP = max(self.log['val_P_acc'])
        self.best_tr_accP = max(self.log['P_acc'])
        self.best_val_accT = max(self.log['val_T_acc'])
        self.best_tr_accT = max(self.log['T_acc'])
        self.best_val_accdT = max(self.log['val_dT_acc'])
        self.best_tr_accdT = max(self.log['dT_acc'])
        self.split = pickle.load(open(self.loadmodelpath+self.corpus+"/split.bp", "rb"))
        self.TBPTT_steps = [int(x) for x in self.split['train'].keys()]

    def saveModel(self, tag=None):
        if tag is not None:
            self.dTmodel.save(self.outpath+tag+"_dTmodel.bp")
            self.Tmodel.save(self.outpath+tag+"_Tmodel.bp")
            self.Pmodel.save(self.outpath+tag+"_Pmodel.bp")
        else:
            self.dTmodel.save(self.outpath+"dTmodel.bp")
            self.Tmodel.save(self.outpath+"Tmodel.bp")
            self.Pmodel.save(self.outpath+"Pmodel.bp")
        pickle.dump(self.log, open(self.outpath+"log.bp", "wb" ))
        pickle.dump(self.split, open(self.outpath+"split.bp", "wb" ))

    def transpose(self):
        for i,XP in enumerate(self.IO['XP']):
            collapsed = np.sum(XP, axis=0)
            active_pitches = np.where(collapsed>0)[0]
            upperbound = active_pitches[-1]
            lowerbound = active_pitches[0]
            possible_shifts = range(0-lowerbound, len(self.dictionaries['P'])-upperbound)
            if len(possible_shifts) == 0:
                possible_shifts.append(0)
            shift = np.random.choice(possible_shifts)
            self.IO['XP'][i] = np.roll(XP, shift, axis=1)
            self.IO['YP'][i] = np.roll(self.IO['YP'][i], shift, axis=1)


    def trainModel(self, epochs=500, validation_split=0.1):

        if self.IO is None:
            self.data2ANN()

        dTvocsize = len(self.dictionaries['dT'])
        Tvocsize = len(self.dictionaries['T'])
        Pvocsize = len(self.dictionaries['P'])

        print('\nTraining Model')

        nb_samples = self.IO['XP'].shape[0]
        all_idxes = np.asarray(range(nb_samples))
        all_batch_idxes = {'train': {}, 'valid': {}}

        if self.split is None:
            idxes_to_add = []
            for step in set(self.TBPTT_steps):
                #print(step)
                idxes = all_idxes[np.where(self.TBPTT_steps==step)]
                np.random.shuffle(idxes)
                idxes = list(idxes)
                if idxes_to_add:
                    idxes.extend(idxes_to_add)
                    idxes_to_add = []
                #print(idxes)
                nb_samples = len(idxes)
                if nb_samples < 10:#not enough samples for this length
                    idxes_to_add = idxes
                    continue
                #print(nb_samples)
                split_idx = int((1 - validation_split) * nb_samples)
                #split_idx -= split_idx % self.batch_size
                all_batch_idxes['train'][str(step)] = idxes[:split_idx]
                #print(all_batch_idxes['train'][str(step)])
                all_batch_idxes['train'][str(step)].extend(
                                            np.random.choice(idxes[:split_idx], 
                                                    self.batch_size - len(all_batch_idxes['train'][str(step)]) % self.batch_size))
                #print(all_batch_idxes['train'][str(step)])
                all_batch_idxes['valid'][str(step)] = idxes[split_idx:]
                all_batch_idxes['valid'][str(step)].extend(
                                            np.random.choice(idxes[split_idx:], 
                                                    self.batch_size - len(all_batch_idxes['valid'][str(step)]) % self.batch_size))

            self.split = all_batch_idxes
        else:
            all_batch_idxes = self.split
            
        
        self.TBPTT_steps = [int(x) for x in self.split['train'].keys()]
        
        
        for epoch in range(epochs):
            tr_epoch_res = []
            self.transpose()
            np.random.shuffle(self.TBPTT_steps)
            for step_number in self.TBPTT_steps:
                #Reshape for batches
                np.random.shuffle(all_batch_idxes['train'][str(step_number)])
                batch_idxes = np.reshape(all_batch_idxes['train'][str(step_number)], (-1,self.batch_size))
                batch_nbr_per_epoch = len(batch_idxes)
                for batch, idxes in enumerate(batch_idxes):

                    # self.Pmodel.reset_states()
                    # for step in range(step_number):
                    #     lossP, accP = self.Pmodel.train_on_batch({'XdT_n': self.IO['XdT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                    #                                 'XT_n': self.IO['XT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                    #                                 'XP_n': self.IO['XP'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size],
                    #                                 'XdT_np1': self.IO['YdT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size],
                    #                                 'XT_np1': self.IO['YT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size]}, 
                    #                                {'YP': self.IO['YP'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size]})


                    self.dTmodel.reset_states()
                    batch_lossdT = []
                    batch_accdT = []
                    for step in range(step_number):
                        lossdT, accdT = self.dTmodel.train_on_batch({'XdT_n': self.IO['XdT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                    'XT_n': self.IO['XT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                    'XP_n': self.IO['XP'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size]}, 
                                                   {'YdT': self.IO['YdT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size]})
                        batch_lossdT.append(lossdT)
                        batch_accdT.append(accdT)

                    self.Tmodel.reset_states()
                    batch_lossT = []
                    batch_accT = []
                    for step in range(step_number):
                        lossT, accT = self.Tmodel.train_on_batch({'XdT_n': self.IO['XdT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                    'XT_n': self.IO['XT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                    'XP_n': self.IO['XP'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size],
                                                    'XdT_np1': self.IO['YdT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size]}, 
                                                   {'YT': self.IO['YT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size]})
                        batch_lossT.append(lossT)
                        batch_accT.append(accT)

                    self.Pmodel.reset_states()
                    batch_lossP = []
                    batch_accP = []
                    for step in range(step_number):
                        lossP, accP = self.Pmodel.train_on_batch({'XdT_n': self.IO['XdT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                    'XT_n': self.IO['XT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                    'XP_n': self.IO['XP'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size],
                                                    'XdT_np1': self.IO['YdT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size],
                                                    'XT_np1': self.IO['YT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size]}, 
                                                   {'YP': self.IO['YP'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size]})
                        batch_lossP.append(lossP)
                        batch_accP.append(accP)

                    lossdT = np.mean(batch_lossdT) 
                    lossT = np.mean(batch_lossT) 
                    lossP = np.mean(batch_lossP) 

                    accdT = np.mean(batch_accdT)
                    accT = np.mean(batch_accT)
                    accP = np.mean(batch_accP)
                    

                    res = [.1*lossdT+.3*lossT+.6*lossP, lossdT, lossT, lossP, accdT, accT, accP] 
                    tr_epoch_res.append(res)

            tr_epoch_res = np.nanmean(tr_epoch_res, axis=0)
            tr_loss, tr_lossdT, tr_lossT, tr_lossP, tr_accdT, tr_accT, tr_accP = tr_epoch_res
            print("Epoch %i/%i\t loss: %.2f - dT: %.2f - T: %.2f - P: %.2f" %
                 (epoch+1,epochs,tr_loss,tr_accdT, tr_accT, tr_accP))
            
            val_epoch_res = []
            for step_number in self.TBPTT_steps:
                #Reshape for batches
                np.random.shuffle(all_batch_idxes['valid'][str(step_number)])
                batch_idxes = np.reshape(all_batch_idxes['valid'][str(step_number)], (-1,self.batch_size))
                batch_nbr_per_epoch = len(batch_idxes)
                for batch, idxes in enumerate(batch_idxes):
                    self.dTmodel.reset_states()
                    batch_lossdT = []
                    batch_accdT = []
                    for step in range(step_number):
                        lossdT, accdT = self.dTmodel.test_on_batch({'XdT_n': self.IO['XdT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                    'XT_n': self.IO['XT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                    'XP_n': self.IO['XP'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size]}, 
                                                   {'YdT': self.IO['YdT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size]})
                        batch_lossdT.append(lossdT)
                        batch_accdT.append(accdT)

                    self.Tmodel.reset_states()
                    batch_lossT = []
                    batch_accT = []
                    for step in range(step_number):   
                        lossT, accT = self.Tmodel.test_on_batch({'XdT_n': self.IO['XdT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                    'XT_n': self.IO['XT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                    'XP_n': self.IO['XP'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size],
                                                    'XdT_np1': self.IO['YdT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size]}, 
                                                   {'YT': self.IO['YT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size]})
                        batch_lossT.append(lossT)
                        batch_accT.append(accT)
                    self.Pmodel.reset_states()
                    batch_lossP = []
                    batch_accP = []
                    for step in range(step_number):
                        lossP, accP = self.Pmodel.test_on_batch({'XdT_n': self.IO['XdT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                    'XT_n': self.IO['XT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size], 
                                                    'XP_n': self.IO['XP'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size],
                                                    'XdT_np1': self.IO['YdT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size],
                                                    'XT_np1': self.IO['YT'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size]}, 
                                                   {'YP': self.IO['YP'][idxes,step*self.TBPTT_size:(step+1)*self.TBPTT_size]})
                        batch_lossP.append(lossP)
                        batch_accP.append(accP)
                    lossdT = np.nanmean(batch_lossdT) 
                    lossT = np.nanmean(batch_lossT) 
                    lossP = np.nanmean(batch_lossP) 

                    accdT = np.nanmean(batch_accdT)
                    accT = np.nanmean(batch_accT)
                    accP = np.nanmean(batch_accP)
                    

                    res = [.1*lossdT+.3*lossT+.6*lossP, lossdT, lossT, lossP, accdT, accT, accP] 
                    val_epoch_res.append(res)
            val_epoch_res = np.nanmean(val_epoch_res, axis=0)
            val_loss, val_lossdT, val_lossT, val_lossP, val_accdT, val_accT, val_accP = val_epoch_res
            print("\t\tloss: %.2f - dT: %.2f - T: %.2f - P: %.2f" %
                 (val_loss,val_accdT, val_accT, val_accP))
            
            self.log['loss'].append(tr_loss)
            self.log['dT_acc'].append(tr_accdT)
            self.log['T_acc'].append(tr_accT)
            self.log['P_acc'].append(tr_accP)
            self.log['dT_loss'].append(tr_lossdT)
            self.log['T_loss'].append(tr_lossT)
            self.log['P_loss'].append(tr_lossP)
            self.log['val_loss'].append(val_loss)
            self.log['val_dT_acc'].append(val_accdT)
            self.log['val_T_acc'].append(val_accT)
            self.log['val_P_acc'].append(val_accP)
            self.log['val_dT_loss'].append(val_lossdT)
            self.log['val_T_loss'].append(val_lossT)
            self.log['val_P_loss'].append(val_lossP)
            
            if  val_accP > self.best_val_accP :
                self.best_val_accP = val_accP
                self.Pmodel.save(self.outpath+"Pmodel.bp")
                pickle.dump(self.log, open(self.outpath+"log.bp", "wb" ))
                pickle.dump(self.split, open(self.outpath+"split.bp", "wb" ))

            if  tr_accP > self.best_tr_accP:
                self.best_tr_accP = tr_accP
                self.Pmodel.save(self.outpath+"train_Pmodel.bp")
                pickle.dump(self.log, open(self.outpath+"log.bp", "wb" ))
                pickle.dump(self.split, open(self.outpath+"split.bp", "wb" ))

            if  val_accT > self.best_val_accT :
                self.best_val_accT = val_accT
                self.Tmodel.save(self.outpath+"Tmodel.bp")
                pickle.dump(self.log, open(self.outpath+"log.bp", "wb" ))
                pickle.dump(self.split, open(self.outpath+"split.bp", "wb" ))

            if  tr_accT > self.best_tr_accT:
                self.best_tr_accT = tr_accT
                self.Tmodel.save(self.outpath+"train_Tmodel.bp")
                pickle.dump(self.log, open(self.outpath+"log.bp", "wb" ))
                pickle.dump(self.split, open(self.outpath+"split.bp", "wb" ))

            if  val_accdT > self.best_val_accdT :
                self.best_val_accdT = val_accdT
                self.dTmodel.save(self.outpath+"dTmodel.bp")
                pickle.dump(self.log, open(self.outpath+"log.bp", "wb" ))
                pickle.dump(self.split, open(self.outpath+"split.bp", "wb" ))

            if  tr_accdT > self.best_tr_accdT:
                self.best_tr_accdT = tr_accdT
                self.dTmodel.save(self.outpath+"train_dTmodel.bp")
                pickle.dump(self.log, open(self.outpath+"log.bp", "wb" ))
                pickle.dump(self.split, open(self.outpath+"split.bp", "wb" ))

    def generate(self, note_len=1000, until_all_ended=True, temperature=0.5, seed=None):


        if until_all_ended:
            note_len = self.IO['XP'].shape[1]

        tag='START/END'
        if tag not in self.dictionaries['dT']:
            self.dictionaries['dT'].append(tag)
            self.dictionaries['T'].append(tag)
            self.dictionaries['P'].append(tag)

        dTvocsize = len(self.dictionaries['dT'])
        Tvocsize = len(self.dictionaries['T'])
        Pvocsize = len(self.dictionaries['P'])
        n_ex = self.batch_size

        ended = [False]*n_ex

        XdTs_hat = np.zeros((n_ex, note_len+1, dTvocsize), dtype = int)
        XdTs_hat[:,0,-1] = 1
        XdTs_probs = np.zeros((n_ex, note_len+1, dTvocsize), dtype = float)

        XTs_hat = np.zeros((n_ex, note_len+1, Tvocsize), dtype = int)
        XTs_hat[:,0,-1] = 1
        XTs_probs = np.zeros((n_ex, note_len+1, Tvocsize), dtype = float)

        XPs_hat = np.zeros((n_ex, note_len+1, Pvocsize), dtype = int)
        XPs_hat[:,0,-1] = 1
        XPs_probs = np.zeros((n_ex, note_len+1, Pvocsize), dtype = float)

        if seed is not None:
            seed_len = len(seed['dT'])
            for i in range(n_ex):
                XdTs_hat[i,1:seed_len+1] = seed['dT']
                XTs_hat[i,1:seed_len+1] = seed['T']
                XPs_hat[i,1:seed_len+1] = seed['P']

        xdt_t = np.zeros((n_ex, self.TBPTT_size, dTvocsize), dtype=int)
        xt_t = np.zeros((n_ex, self.TBPTT_size, Tvocsize), dtype=int)
        xp_t = np.zeros((n_ex, self.TBPTT_size, Pvocsize), dtype=int)
        xdt_tp1 = np.zeros((n_ex, self.TBPTT_size, dTvocsize), dtype=int)
        xt_tp1 = np.zeros((n_ex, self.TBPTT_size, Tvocsize), dtype=int)

        self.dTmodel.reset_states()
        self.Tmodel.reset_states()
        self.Pmodel.reset_states()

        try:
            last_note = 0
            for t in tqdm(range(note_len)):  
                xdt_t[:,0] = XdTs_hat[:,t]
                xt_t[:,0] = XTs_hat[:,t]
                xp_t[:,0] = XPs_hat[:,t]

                probs = self.dTmodel.predict([xdt_t, xt_t, xp_t])
                XdTs_probs[:,t] = probs[:, 0]  
                for idx in range(n_ex):  
                    dT_np1 = utils.sample(probs[idx, 0], temperature=temperature)
                    if np.sum(XdTs_hat[idx, t+1]) == 0:
                        XdTs_hat[idx, t+1, dT_np1] = 1
                        if self.dictionaries['dT'][dT_np1] == 'START/END':
                            if ended[idx] == False:
                                print("%i dT ended @note #%i"%(idx+1, t))
                            ended[idx] = True 

                xdt_tp1[:,0] = XdTs_hat[:,t+1]

                probs = self.Tmodel.predict([xdt_t, xt_t, xp_t, xdt_tp1])
                XTs_probs[:,t] = probs[:, 0]          
                for idx in range(n_ex):  
                    T_np1 = utils.sample(probs[idx, 0], temperature=temperature)
                    if np.sum(XTs_hat[idx, t+1]) == 0:
                        XTs_hat[idx, t+1, T_np1] = 1
                        if self.dictionaries['T'][T_np1] == 'START/END':
                            if ended[idx] == False:
                                print("%i T ended @note #%i"%(idx+1, t))
                            ended[idx] = True 
                xt_tp1[:,0] = XTs_hat[:,t+1]

                probs = self.Pmodel.predict([xdt_t, xt_t, xp_t, xdt_tp1, xt_tp1])
                for idx in range(n_ex):  
                    if not ended[idx]:
                        XPs_probs[idx,t] = probs[idx, 0]
                    if np.sum(XPs_hat[idx, t+1]) == 0:
                        P_np1 = utils.sample(probs[idx, 0], temperature=temperature)
                        XPs_hat[idx, t+1, P_np1] = 1.
                        if self.dictionaries['P'][P_np1] == 'START/END':
                            if ended[idx] == False:
                                print("%i P ended @note #%i"%(idx+1, t))
                            ended[idx] = True 



                last_note = t
                if until_all_ended == True and np.sum(ended) == n_ex:
                    break
        except KeyboardInterrupt:
            print("ctrl-c ended @note %i"%(t))
            pass
        print("End generating: %i/%i song ended"%(np.sum(ended),n_ex))
        return [XdTs_hat[:,:last_note], XTs_hat[:,:last_note], XPs_hat[:,:last_note], XdTs_probs[:,:last_note], XTs_probs[:,:last_note], XPs_probs[:,:last_note]]


if __name__ == "__main__":

    m = BachProp(sys.argv[1])
    
    m.loadData()
    m.data2ANN()
    if sys.argv[2] == "load":
        m.loadModel()
    else:
        m.buildModel()
        m.trainModel()
    
    XdTs_hat, XTs_hat, XPs_hat, XdTs_probs, XTs_probs, XPs_probs = m.generate()
    dTs, Ts, Ps = m.ANN2data(XdTs_hat, XTs_hat, XPs_hat)
    utils.longMIDI(dTs, Ts, Ps, path=m.genpath, label='generated', bpm=90)


