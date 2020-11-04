# -*- coding: utf-8 -*-

import argparse

from supar import BiaffineDependencyParser
from supar.cmds.cmd import parse


def main():
    parser = argparse.ArgumentParser(description='Create Biaffine Dependency Parser.')
    parser.add_argument('--tree', action='store_true', help='whether to ensure well-formedness')
    parser.add_argument('--proj', action='store_true', help='whether to projectivise the data')
    parser.add_argument('--partial', action='store_true', help='whether partial annotation is included')
    parser.set_defaults(Parser=BiaffineDependencyParser)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', choices=['tag', 'char', 'bert', 'elmo'], help='choices of additional features')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--train', default='data/ptb/train.conllx', help='path to train file')
    subparser.add_argument('--dev', default='data/ptb/dev.conllx', help='path to dev file')
    subparser.add_argument('--test', default='data/ptb/test.conllx', help='path to test file')
    subparser.add_argument('--embed', default='data/glove.6B.100d.txt', help='path to pretrained embeddings')
    subparser.add_argument('--unk', default='unk', help='unk token in pretrained embeddings')
    subparser.add_argument('--n-embed', default=1024, type=int, help='dimension of embeddings')
    subparser.add_argument('--bert', default='bert-base-cased', help='which bert model to use')
    subparser.add_argument('--epochs', default=5000, type=int)
    subparser.add_argument('--elmo_weights')
    subparser.add_argument('--elmo_options')
    subparser.add_argument('--map-layer0', help='path to mapping model for layer0')
    subparser.add_argument('--map-layer1', help='path to mapping model for layer1')
    subparser.add_argument('--map-layer2', help='path to mapping model for layer2')
    subparser.add_argument('--map-method', choices['vecmap', 'elmogan', 'none'], default='none')
    subparser.add_argument('--vecmap-lang', help='was data language source(src) or target(trg) during mapping (vecmap only)')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/ptb/test.conllx', help='path to dataset')
    subparser.add_argument('--elmo_weights')
    subparser.add_argument('--elmo_options')
    subparser.add_argument('--map-layer0', help='path to mapping model for layer0')
    subparser.add_argument('--map-layer1', help='path to mapping model for layer1')
    subparser.add_argument('--map-layer2', help='path to mapping model for layer2')
    subparser.add_argument('--map-method', choices['vecmap', 'elmogan', 'none'], default='none')
    subparser.add_argument('--map-direction', choices=[0,1], help='which direction to map, xx-yy_to_yy-xx, 0 implies xx->yy, 1 implies yy->xx (elmogan only)')
    subparser.add_argument('--vecmap-lang', help='was data language source(src) or target(trg) during mapping (vecmap only)')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/ptb/test.conllx', help='path to dataset')
    subparser.add_argument('--pred', default='pred.conllx', help='path to predicted result')
    subparser.add_argument('--elmo_weights')
    subparser.add_argument('--elmo_options')
    subparser.add_argument('--map-layer0', help='path to mapping model for layer0')
    subparser.add_argument('--map-layer1', help='path to mapping model for layer1')
    subparser.add_argument('--map-layer2', help='path to mapping model for layer2')
    subparser.add_argument('--map-method', choices['vecmap', 'elmogan', 'none'], default='none')
    subparser.add_argument('--map-direction', choices=[0,1], help='which direction to map, xx-yy_to_yy-xx, 0 implies xx->yy, 1 implies yy->xx (elmogan only)')
    subparser.add_argument('--vecmap-lang', help='was data language source(src) or target(trg) during mapping (vecmap only)')
    parse(parser)


if __name__ == "__main__":
    main()
