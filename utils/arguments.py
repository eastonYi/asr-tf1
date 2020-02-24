import logging
import sys
import os
import yaml
import shutil
from pathlib import Path
from argparse import ArgumentParser
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')

from .dataProcess import load_vocab
from models.utils.tfData import TFDataReader
from .tools import mkdirs, AttrDict

parser = ArgumentParser()
parser.add_argument('-m', type=str, dest='mode', default='train')
parser.add_argument('--name', type=str, dest='name', default=None)
parser.add_argument('--gpu', type=str, dest='gpu', default=None)
parser.add_argument('--c', type=str, dest='config')

param = parser.parse_args()

CONFIG_FILE = sys.argv[-1]
# CONFIG_FILE = param.config
args = AttrDict(yaml.load(open(CONFIG_FILE), Loader=yaml.SafeLoader))

args.mode = param.mode

args.gpus = param.gpu if param.gpu else args.gpus
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
args.num_gpus = len(args.gpus.split(','))
args.list_gpus = ['/gpu:{}'.format(i) for i in range(args.num_gpus)]

# bucket
if args.bucket_boundaries:
    args.list_bucket_boundaries = [int(i) for i in args.bucket_boundaries.split(',')]

assert args.num_batch_tokens
args.list_batch_size = ([int(args.num_batch_tokens / boundary) * args.num_gpus
        for boundary in (args.list_bucket_boundaries)] + [args.num_gpus])
args.batch_size *= args.num_gpus
logging.info('\nbucket_boundaries: {} \nbatch_size: {}'.format(
    args.list_bucket_boundaries, args.list_batch_size))

# dirs
dir_dataInfo = Path.cwd() / 'data'
dir_exps = Path.cwd() / 'exps' / args.dirs.exp
args.dir_exps = dir_exps / CONFIG_FILE.split('/')[-1].split('.')[0]
if param.name:
    args.dir_exps = args.dir_exps / param.name
args.dir_log = args.dir_exps / 'log'
args.dir_checkpoint = args.dir_exps / 'checkpoint'
args.dirs.train.tfdata = Path(args.dirs.train.tfdata) if args.dirs.train.tfdata else None
args.dirs.dev.tfdata = Path(args.dirs.dev.tfdata) if args.dirs.dev.tfdata else None
args.dirs.test.tfdata = Path(args.dirs.test.tfdata) if args.dirs.test.tfdata else None

mkdirs(dir_dataInfo)
mkdirs(dir_exps)
mkdirs(args.dir_exps)
mkdirs(args.dir_log)
mkdirs(args.dir_checkpoint)

shutil.copy(CONFIG_FILE, str(args.dir_exps))

# vocab
args.token2idx, args.idx2token = load_vocab(args.dirs.vocab)
if args.dirs.vocab_phone:
    args.phone2idx, args.idx2phone = load_vocab(args.dirs.vocab_phone)
args.dim_output = len(args.token2idx)

args.eos_idx = args.token2idx['<eos>']
args.sos_idx = args.token2idx['<sos>']

args.dirs.train.tfdata = Path(args.dirs.train.tfdata)
args.dirs.dev.tfdata = Path(args.dirs.dev.tfdata)
args.dirs.test.tfdata = Path(args.dirs.test.tfdata)
args.dirs.train.feat_len = args.dirs.train.tfdata/'feature_length.txt'
args.dirs.dev.feat_len = args.dirs.dev.tfdata/'feature_length.txt'

assert args.sample_uttid

if args.dirs.type == 'scp':
    from .dataset import ASR_scp_DataSet
    dataset_train = ASR_scp_DataSet(
        f_scp=args.dirs.train.scp,
        f_trans=args.dirs.train.trans,
        args=args,
        _shuffle=True,
        transform=False)
    dataset_dev = ASR_scp_DataSet(
        f_scp=args.dirs.dev.scp,
        f_trans=args.dirs.dev.trans,
        args=args,
        _shuffle=False,
        transform=False)
    dataset_test = ASR_scp_DataSet(
        f_scp=args.dirs.test.scp,
        f_trans=args.dirs.test.trans,
        args=args,
        _shuffle=False,
        transform=True)
elif args.dirs.type == 'scp_multi':
    from .dataset import ASR_phone_char_ArkDataSet
    dataset_train = ASR_phone_char_ArkDataSet(
        f_scp=args.dirs.train.scp,
        f_phone=args.dirs.train.phone,
        f_char=args.dirs.train.char,
        args=args,
        _shuffle=True,
        transform=False)
    dataset_dev = ASR_phone_char_ArkDataSet(
        f_scp=args.dirs.dev.scp,
        f_phone=args.dirs.dev.phone,
        f_char=args.dirs.dev.char,
        args=args,
        _shuffle=False,
        transform=False)
    dataset_test = ASR_phone_char_ArkDataSet(
        f_scp=args.dirs.test.scp,
        f_phone=args.dirs.test.phone,
        f_char=args.dirs.test.char,
        args=args,
        _shuffle=False,
        transform=True)
else :
    dataset_dev = dataset_train = dataset_test = None

args.dataset_dev = dataset_dev
args.dataset_train = dataset_train
args.dataset_test = dataset_test

try:
    args.data.dim_feature = TFDataReader.read_tfdata_info(args.dirs.train.tfdata)['dim_feature']
    args.data.train_size = TFDataReader.read_tfdata_info(args.dirs.train.tfdata)['size_dataset']
    args.data.dev_size = TFDataReader.read_tfdata_info(args.dirs.dev.tfdata)['size_dataset']
    args.data.dim_input = args.data.dim_feature * \
            (args.data.right_context + args.data.left_context +1) *\
            (3 if args.data.add_delta else 1)
except:
    print("have not converted to tfdata yet: ")

# model
## encoder
if args.model.encoder.type == 'transformer_encoder':
    from models.encoders.transformer_encoder import Transformer_Encoder as encoder
elif args.model.decoder.type == 'conv_transformer_encoder':
    from models.decoders.transformer_encoder import Conv_Transformer_Encoder as encoder
elif args.model.encoder.type == 'conv_lstm':
    from models.encoders.conv_lstm import CONV_LSTM as encoder
elif args.model.encoder.type == 'blstm':
    from models.encoders.blstm import BLSTM as encoder
elif args.model.encoder.type == 'conv_1d':
    from models.encoders.conv import CONV_1D as encoder
elif args.model.encoder.type == 'conv_2d':
    from models.encoders.conv import CONV_2D as encoder
else:
    raise NotImplementedError('not found encoder type: {}'.format(args.model.encoder.type))
args.model.encoder.type = encoder

## decoder
if args.model.decoder.type == 'FC':
    from models.decoders.fc_decoder import FCDecoder as decoder
elif args.model.decoder.type == 'conv_decoder':
    from models.decoders.conv_decoder import CONV_Decoder as decoder
elif args.model.decoder.type == 'transformer_decoder':
    from models.decoders.transformer_decoder import Transformer_Decoder as decoder
else:
    raise NotImplementedError('not found decoder type: {}'.format(args.model.decoder.type))
args.model.decoder.type = decoder

## model
if args.model.type == 'Seq2SeqModel':
    from models.seq2seqModel import Seq2SeqModel as Model
elif args.model.type == 'ctcModel':
    from models.ctcModel import CTCModel as Model
elif args.model.type == 'Ectc_Docd':
    from models.Ectc_Docd import Ectc_Docd as Model
elif args.model.type == 'Ectc_Docd_Multi':
    from models.Ectc_Docd import Ectc_Docd_Multi as Model
elif args.model.type == 'Ectc_Docd_Multi_2En':
    from models.Ectc_Docd import Ectc_Docd_Multi_2En as Model
    if args.model.encoder2.type == 'conv_lstm':
        from models.encoders.conv_lstm import CONV_LSTM as encoder
    elif args.model.encoder2.type == 'conv_lstm_4x':
        from models.encoders.conv_lstm import CONV_LSTM_4x as encoder
    elif args.model.encoder2.type == 'blstm':
        from models.encoders.blstm import BLSTM as encoder
    elif args.model.encoder2.type == 'conv_1d':
        from models.encoders.conv import CONV_1D as encoder
    elif args.model.encoder2.type == 'conv_2d':
        from models.encoders.conv import CONV_2D as encoder
    elif args.model.encoder2.type == 'conv_1d_rnn':
        from models.encoders.conv import CONV_1D_with_RNN as encoder
    else:
        raise NotImplementedError('not found encoder type: {}'.format(args.model.encoder2.type))
    args.model.encoder2.type = encoder

elif args.model.type == 'transformer':
    from models.transformer import Transformer as Model
else:
    raise NotImplementedError('not found Model type!')

args.Model = Model

if args.model_D:
    if args.model_D.type == 'clm':
        from models.discriminator.clm import CLM as Model_D
    else:
        raise NotImplementedError('not found Model type!')
    args.Model_D = Model_D

    from models.gan import GAN_2 as GAN
    args.GAN = GAN
