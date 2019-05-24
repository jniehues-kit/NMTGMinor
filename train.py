from __future__ import division

import onmt
import onmt.Markdown
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time, datetime
from onmt.train_utils.trainer import XETrainer
from onmt.train_utils.fp16_trainer import FP16XETrainer
from onmt.train_utils.multiGPUtrainer import MultiGPUXETrainer
from onmt.modules.Loss import NMTLossFunc, NMTAndCTCLossFunc,NMTAndPOSLossFunc
from onmt.ModelConstructor import build_model, init_model_parameters

parser = argparse.ArgumentParser(description='train.py')
onmt.Markdown.add_md_help_argument(parser)

from options import make_parser
# Please look at the options file to see the options regarding models and data
parser = make_parser(parser)

opt = parser.parse_args()

print(opt)

# An ugly hack to have weight norm on / off
onmt.Constants.weight_norm = opt.weight_norm
onmt.Constants.checkpointing = opt.checkpointing
onmt.Constants.max_position_length = opt.max_position_length

# Use static dropout if checkpointing > 0
if opt.checkpointing > 0:
    onmt.Constants.static = True

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")


torch.manual_seed(opt.seed)


def main():

    if opt.data_format == 'raw':
        start = time.time()
        if opt.data.endswith(".train.pt"):
            print("Loading data from '%s'" % opt.data)
            dataset = torch.load(opt.data)
        else:
            print("Loading data from %s" % opt.data + ".train.pt")
            dataset = torch.load(opt.data + ".train.pt")

        elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
        print("Done after %s" % elapse )


        train_data = onmt.Dataset(dataset['train']['src'],
                                 dataset['train']['tgt'], opt.batch_size_words,
                                 data_type=dataset.get("type", "text"),
                                 batch_size_sents=opt.batch_size_sents,
                                 multiplier = opt.batch_size_multiplier)
        valid_data = onmt.Dataset(dataset['valid']['src'],
                                 dataset['valid']['tgt'], opt.batch_size_words,
                                 data_type=dataset.get("type", "text"),
                                 batch_size_sents=opt.batch_size_sents)

        dicts = dataset['dicts']
        if "src" in dicts:
            print(' * vocabulary size. source = %d; target = %d' %
            (dicts['src'].size(), dicts['tgt'].size()))
        else:
            print(' * vocabulary size. target = %d' %
            (dicts['tgt'].size()))

        print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
        print(' * maximum batch size (words per batch). %d' % opt.batch_size_words)

    elif opt.data_format == 'bin':

        from onmt.data_utils.IndexedDataset import IndexedInMemoryDataset

        dicts = torch.load(opt.data + ".dict.pt")

        #~ train = {}
        train_path = opt.data + '.train'
        train_src = IndexedInMemoryDataset(train_path + '.src')
        train_tgt = IndexedInMemoryDataset(train_path + '.tgt')

        train_data = onmt.Dataset(train_src,
                                 train_tgt, opt.batch_size_words,
                                 data_type=opt.encoder_type,
                                 batch_size_sents=opt.batch_size_sents,
                                 multiplier = opt.batch_size_multiplier)

        valid_path = opt.data + '.valid'
        valid_src = IndexedInMemoryDataset(valid_path + '.src')
        valid_tgt = IndexedInMemoryDataset(valid_path + '.tgt')

        valid_data = onmt.Dataset(valid_src,
                                 valid_tgt, opt.batch_size_words,
                                 data_type=opt.encoder_type,
                                 batch_size_sents=opt.batch_size_sents)

    else:
        raise NotImplementedError

    print('Building model...')

    if not opt.fusion:
        model = build_model(opt, dicts)

        """ Building the loss function """
        if opt.ctc_loss != 0:
            loss_function = NMTAndCTCLossFunc(dicts['tgt'].size(), label_smoothing=opt.label_smoothing,ctc_weight = opt.ctc_loss)
        elif opt.predict_position == "relative":
            loss_function = NMTAndPOSLossFunc(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)
        else:
            loss_function = NMTLossFunc(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)
    else:
        from onmt.ModelConstructor import build_fusion
        from onmt.modules.Loss import FusionLoss

        model = build_fusion(opt, dicts)

        loss_function = FusionLoss(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)


    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)

    if len(opt.gpus) > 1 or opt.virtual_gpu > 1:
        #~ trainer = MultiGPUXETrainer(model, loss_function, train_data, valid_data, dataset, opt)
        raise NotImplementedError("Warning! Multi-GPU training is not fully tested and potential bugs can happen.")
    else:
        if opt.fp16:
            trainer = FP16XETrainer(model, loss_function, train_data, valid_data, dicts, opt)
        else:
            trainer = XETrainer(model, loss_function, train_data, valid_data, dicts, opt)

    
    trainer.run(save_file=opt.load_from)

if __name__ == "__main__":
    main()
