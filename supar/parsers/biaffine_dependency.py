# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from supar.models import BiaffineDependencyModel
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import bos, pad, unk
from supar.utils.field import Field, SubwordField, ElmoField
from supar.utils.fn import ispunct
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import AttachmentMetric
from supar.utils.transform import CoNLL
from allennlp.commands.elmo import ElmoEmbedder
from supar.xlingual.elmo_mapper import Elmogan, Vecmap, Muse
from elmoformanylangs import Embedder as EFML
logger = get_logger(__name__)


class BiaffineDependencyParser(Parser):
    r"""
    The implementation of Biaffine Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning. 2017.
          `Deep Biaffine Attention for Neural Dependency Parsing`_.

    .. _Deep Biaffine Attention for Neural Dependency Parsing:
        https://openreview.net/forum?id=Hk95PK9le
    """

    NAME = 'biaffine-dependency'
    MODEL = BiaffineDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.args.feat in ('char', 'bert', 'elmo'):
            self.WORD, self.FEAT = self.transform.FORM
        else:
            self.WORD, self.FEAT = self.transform.FORM, self.transform.CPOS
        self.ARC, self.REL = self.transform.HEAD, self.transform.DEPREL
        self.puncts = torch.tensor([i
                                    for s, i in self.WORD.vocab.stoi.items()
                                    if ispunct(s)]).to(self.args.device)
        if self.args.elmo_options:
            self.elmo = ElmoEmbedder(self.args.elmo_options, self.args.elmo_weights, -1)
        else:
            self.efml = EFML(self.args.elmo_weights)
            self.elmo = False
        #print(self.__dict__)
        if self.args.map_method == 'vecmap':
            self.mapper = Vecmap(vars(self.args))
        elif self.args.map_method == 'elmogan':
            self.mapper = Elmogan(vars(self.args))
        elif self.args.map_method == 'muse':
            self.mapper = Muse(vars(self.args))
        else:
            self.mapper = None
            
    def train(self, train, dev, test, buckets=32, batch_size=5000,
              punct=False, tree=False, proj=False, partial=False, verbose=True, **kwargs):
        r"""
        Args:
            train/dev/test (list[list] or str):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            punct (bool):
                If ``False``, ignores the punctuations during evaluation. Default: ``False``.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for training.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000,
                 punct=False, tree=True, proj=False, partial=False, verbose=True, **kwargs):
        r"""
        Args:
            data (str):
                The data for evaluation, both list of instances and filename are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            punct (bool):
                If ``False``, ignores the punctuations during evaluation. Default: ``False``.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for evaluation.

        Returns:
            The loss scalar and evaluation results.
        """
        if kwargs['elmo_options']:
            self.elmo = ElmoEmbedder(kwargs['elmo_options'], kwargs['elmo_weights'], -1)
        else:
            self.efml = EFML(kwargs['elmo_weights'])
        if kwargs['map_method'] == 'vecmap':
            self.mapper = Vecmap(kwargs)
            #print(self.mapper)
        elif kwargs['map_method'] == 'elmogan':
            self.mapper = Elmogan(kwargs)
            #print(self.mapper)
        elif kwargs['map_method'] == 'muse':
            self.mapper = Muse(kwargs)
        else:
            self.mapper = None

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, buckets=8, batch_size=5000,
                prob=False, tree=True, proj=False, verbose=True, **kwargs):
        r"""
        Args:
            data (list[list] or str):
                The data for prediction, both a list of instances and filename are allowed.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for prediction.

        Returns:
            A :class:`~supar.utils.Dataset` object that stores the predicted results.
        """
        
        if kwargs['elmo_options']:
            self.elmo = ElmoEmbedder(kwargs['elmo_options'], kwargs['elmo_weights'], -1)
        else:
            self.efml = EFML(kwargs['elmo_weights'])
        if kwargs['map_method'] == 'vecmap':
            self.mapper = Vecmap(kwargs)
            #print(self.mapper)
        elif kwargs['map_method'] == 'elmogan':
            self.mapper = Elmogan(kwargs)
            #print(self.mapper)
        elif kwargs['map_method'] == 'muse':
            self.mapper = Muse(kwargs)
        else:
            self.mapper = None
        return super().predict(**Config().update(locals()))

    def _train(self, loader):
        self.model.train()

        bar, metric = progress_bar(loader), AttachmentMetric()
        # words, feats, etc. come from loader! loader is train.loader, where train is Dataset
        for words, feats, arcs, rels in bar:
            self.optimizer.zero_grad()
            if self.elmo:
                feat_embs = self.elmo.embed_batch(feats)
            else:
                feat_embs = self.efml.sents2elmo(feats, output_layer=-2)
            #TODO: dodaj mapping, ce in samo ce gre za vecmap
            if self.args.map_method == 'vecmap':
                # map feat_embs with vecmap, actually self.mapper defined in class init
                feat_embs = self.mapper.map_batch(feat_embs)
                
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            
            feats0 = torch.zeros(words.shape+(1024,)) # words.clone()
            feats1 = torch.zeros(words.shape+(1024,))
            feats2 = torch.zeros(words.shape+(1024,))
            # words get ignored, all input comes from feats - 3 elmo layers
            # still inputting words due to reasons(tm)
            
            #feats0 = feats0.unsqueeze(-1)
            #feats0 = feats0.expand(words.shape+(1024,))
            for sentence in range(len(feat_embs)):
                for token in range(len(feat_embs[sentence][1])):
                    feats0[sentence][token] = torch.Tensor(feat_embs[sentence][0][token])
                    feats1[sentence][token] = torch.Tensor(feat_embs[sentence][1][token])
                    feats2[sentence][token] = torch.Tensor(feat_embs[sentence][2][token])
            feats = torch.cat((feats0, feats1, feats2), -1)
            if str(self.args.device) == '-1':
                feats = feats.to('cpu')
            else:
                feats = feats.to('cuda:'+str(self.args.device)) #TODO: fix to allow cpu or gpu
            s_arc, s_rel = self.model(words, feats) #INFO: here is the data input, y = model(x)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, self.args.partial)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            metric(arc_preds, rel_preds, arcs, rels, mask)
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}")

    @torch.no_grad()
    def _evaluate(self, loader):
        print("called _evaluate function")
        print(self.mapper)
        self.model.eval()

        total_loss, metric = 0, AttachmentMetric()

        for words, feats, arcs, rels in loader:
            if self.elmo:
                feat_embs0 = self.elmo.embed_batch(feats)
            else:
                feat_embs0 = self.efml.sents2elmo(feats, output_layer=-2)
            if self.mapper:
                # map feat_embs with self.mapper defined in class init
                feat_embs = self.mapper.map_batch(feat_embs0)
            else:
                feat_embs = feat_embs0
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            feats0 = torch.zeros(words.shape+(1024,))
            feats1 = torch.zeros(words.shape+(1024,))
            feats2 = torch.zeros(words.shape+(1024,))
            for sentence in range(len(feat_embs)):
                for token in range(len(feat_embs[sentence][1])):
                    feats0[sentence][token] = torch.Tensor(feat_embs[sentence][0][token])
                    feats1[sentence][token] = torch.Tensor(feat_embs[sentence][1][token])
                    feats2[sentence][token] = torch.Tensor(feat_embs[sentence][2][token])
            feats = torch.cat((feats0, feats1, feats2), -1)
            if str(self.args.device) == '-1':
                feats = feats.to('cpu')
            else:
                feats = feats.to('cuda:'+str(self.args.device))
            s_arc, s_rel = self.model(words, feats)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, self.args.partial)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask,
                                                     self.args.tree,
                                                     self.args.proj)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            total_loss += loss.item()
            metric(arc_preds, rel_preds, arcs, rels, mask)
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {}
        arcs, rels, probs = [], [], []
        for words, feats in progress_bar(loader):
            if self.elmo:
                feat_embs = self.elmo.embed_batch(feats)
            else:
                feat_embs = self.efml.sents2elmo(feats, output_layer=-2)
            if self.mapper:
                # map feat_embs with self.mapper defined in class init
                feat_embs = self.mapper.map_batch(feat_embs)
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            feats0 = torch.zeros(words.shape+(1024,))
            feats1 = torch.zeros(words.shape+(1024,))
            feats2 = torch.zeros(words.shape+(1024,))
            for sentence in range(len(feat_embs)):
                for token in range(len(feat_embs[sentence][1])):
                    feats0[sentence][token] = torch.Tensor(feat_embs[sentence][0][token])
                    feats1[sentence][token] = torch.Tensor(feat_embs[sentence][1][token])
                    feats2[sentence][token] = torch.Tensor(feat_embs[sentence][2][token])
            feats = torch.cat((feats0, feats1, feats2), -1)
            if str(self.args.device) == '-1':
                feats = feats.to('cpu')
            else:
                feats = feats.to('cuda:'+str(self.args.device))
            s_arc, s_rel = self.model(words, feats)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask,
                                                     self.args.tree,
                                                     self.args.proj)
            arcs.extend(arc_preds[mask].split(lens))
            rels.extend(rel_preds[mask].split(lens))
            if self.args.prob:
                arc_probs = s_arc.softmax(-1)
                probs.extend([prob[1:i+1, :i+1].cpu() for i, prob in zip(lens, arc_probs.unbind())])
        arcs = [seq.tolist() for seq in arcs]
        rels = [self.REL.vocab[seq.tolist()] for seq in rels]
        preds = {'arcs': arcs, 'rels': rels}
        if self.args.prob:
            preds['probs'] = probs

        return preds

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
        if args.feat == 'char':
            FEAT = SubwordField('chars', pad=pad, unk=unk, bos=bos, fix_len=args.fix_len)
        elif args.feat == 'bert':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.bert)
            FEAT = SubwordField('bert',
                                pad=tokenizer.pad_token,
                                unk=tokenizer.unk_token,
                                bos=tokenizer.bos_token or tokenizer.cls_token,
                                fix_len=args.fix_len,
                                tokenize=tokenizer.tokenize)
            FEAT.vocab = tokenizer.get_vocab()
        elif args.feat == 'elmo':
            logger.info("Hello, initing ElmoField")
            FEAT = ElmoField('elmo', bos=bos) #
        else:
            FEAT = Field('tags', bos=bos)
        ARC = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
        REL = Field('rels', bos=bos)
        if args.feat in ('char', 'bert'):
            transform = CoNLL(FORM=(WORD, FEAT), HEAD=ARC, DEPREL=REL)
        elif args.feat == 'elmo':
            logger.info("calling CoNLL transform")
            # FEAT ima se kar 3 layerje, to bo za popravit nekak
            transform = CoNLL(FORM=(WORD, FEAT), HEAD=ARC, DEPREL=REL)
        else:
            transform = CoNLL(FORM=WORD, CPOS=FEAT, HEAD=ARC, DEPREL=REL)
        logger.info("initing train Dataset")
        train = Dataset(transform, args.train)
        #WORD.build(train, args.min_freq, (Embedding.load(args.embed, args.unk) if args.embed else None))
        logger.info("Building WORD, FEAT, REL fields")
        WORD.build(train)
        FEAT.build(train)
        REL.build(train)
        args.update({
            'n_words': WORD.vocab.n_init,
            'n_feats': len(FEAT.vocab),
            'n_rels': len(REL.vocab),
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            'feat_pad_index': FEAT.pad_index,
        })
        logger.info("Loading model")
        model = cls.MODEL(**args)
        model.load_pretrained(WORD.embed).to(args.device)
        return cls(args, model, transform)
