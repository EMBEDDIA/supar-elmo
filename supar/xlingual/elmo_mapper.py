import tensorflow
from tensorflow.keras.models import Model, load_model
from supar.xlingual.apply_vecmap_transform import vecmap
import numpy as np


class Elmogan():
    def __init__(self, args):
        self.layer0 = load_model(args.map_layer0)
        self.layer1 = load_model(args.map_layer1)
        self.layer2 = load_model(args.map_layer2)
        self.direction = args.map_direction
        
    def map_batch(self, batch):
        for x,sentence in enumerate(batch):
            seqlen = sentence[0].shape[0]
            batch[x, 0, 0:seqlen] = self.apply_mapping(sentence[0], self.layer0)
            batch[x, 1, 0:seqlen] = self.apply_mapping(sentence[1], self.layer1)
            batch[x, 2, 0:seqlen] = self.apply_mapping(sentence[2], self.layer2)
        return batch
            
    def apply_mapping(self, sentence, W):
        if W:
            if self.direction == 0:
                input = [sentence, sentence]
                mapped_sentence, _ = W.predict(input)
            else:
                input = [sentence, sentence]
                _, mapped_sentence = W.predict(input)
        else:
            mapped_sentence = sentence

        return mapped_sentence
    
class Vecmap():
    def __init__(self, args):
        self.layer0 = np.load(args.map_layer0)
        self.layer1 = np.load(args.map_layer1)
        self.layer2 = np.load(args.map_layer2)
        if args.vecmap_lang in ('source', 'src'):
            self.lang = 'wx2'
        elif args.vecmap_lang in ('target', 'trg', 'tgt'):
            self.lang = 'wz2'
        else:
            self.lang = None
    
    def map_batch(self, batch):    
        for x,sentence in enumerate(batch):
            seqlen = sentence[0].shape[0]
            batch[x][0][0:seqlen] = self.apply_mapping(sentence[0], self.layer0)
            batch[x][1][0:seqlen] = self.apply_mapping(sentence[1], self.layer1)
            batch[x][2][0:seqlen] = self.apply_mapping(sentence[2], self.layer2)
        return batch
    
    def apply_mapping(self, sentence, W):
        if W:
            mapped_sentence = vecmap(sentence, W[self.lang], W['s'])
        else:
            mapped_sentence = sentence
        return mapped_sentence
