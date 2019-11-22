import logging
import sys
import yaml
from pathlib import Path
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')

from .dataProcess import load_vocab
from models.utils.tfData import TFData
from .tools import mkdirs, AttrDict


CONFIG_FILE = sys.argv[-1]
args = AttrDict(yaml.load(open(CONFIG_FILE), Loader=yaml.SafeLoader))

args.num_gpus = len(args.gpus.split(','))
args.list_gpus = ['/gpu:{}'.format(i) for i in range(args.num_gpus)]

# bucket
if args.bucket_boundaries:
    args.list_bucket_boundaries = [int(i) for i in args.bucket_boundaries.split(',')]

assert args.num_batch_tokens
args.list_batch_size = ([int(args.num_batch_tokens / boundary) * args.num_gpus
        for boundary in (args.list_bucket_boundaries)] + [args.num_gpus])
args.batch_size *= args.num_gpus
args.text_batch_size *= args.num_gpus
logging.info('\nbucket_boundaries: {} \nbatch_size: {}'.format(
    args.list_bucket_boundaries, args.list_batch_size))

# dirs
dir_dataInfo = Path.cwd() / 'data'
dir_exps = Path.cwd() / 'exps' / args.dirs.exp
args.dir_exps = dir_exps / CONFIG_FILE.split('/')[-1].split('.')[0]
args.dir_log = args.dir_exps / 'log'
args.dir_checkpoint = args.dir_exps / 'checkpoint'
args.dirs.train.tfdata = Path(args.dirs.train.tfdata) if args.dirs.train.tfdata else None
args.dirs.dev.tfdata = Path(args.dirs.dev.tfdata) if args.dirs.dev.tfdata else None

if not dir_dataInfo.is_dir(): dir_dataInfo.mkdir()
if not dir_exps.is_dir(): dir_exps.mkdir()
if not args.dir_exps.is_dir(): args.dir_exps.mkdir()
if not args.dir_log.is_dir(): args.dir_log.mkdir()
if not args.dir_checkpoint.is_dir(): args.dir_checkpoint.mkdir()

# vocab
args.token2idx, args.idx2token = load_vocab(args.dirs.vocab)
args.dim_output = len(args.token2idx)
if '<eos>' in args.token2idx.keys():
    args.eos_idx = args.token2idx['<eos>']
else:
    args.eos_idx = None

if '<sos>' in args.token2idx.keys():
    args.sos_idx = args.token2idx['<sos>']
elif '<blk>' in args.token2idx.keys():
    args.sos_idx = args.token2idx['<blk>']
else:
    args.sos_idx = None

args.dirs.train.tfdata = Path(args.dirs.train.tfdata)
args.dirs.dev.tfdata = Path(args.dirs.dev.tfdata)
try:
    mkdirs(args.dirs.train.tfdata)
    mkdirs(args.dirs.dev.tfdata)
except:
    pass
args.dirs.train.feat_len = args.dirs.train.tfdata/'feature_length.txt'
args.dirs.dev.feat_len = args.dirs.dev.tfdata/'feature_length.txt'

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
elif args.dirs.type == 'csv':
    from .dataset import ASR_csv_DataSet
    dataset_train = ASR_csv_DataSet(
        list_files=args.dirs.train.list_files,
        args=args,
        _shuffle=True,
        transform=False)
    dataset_dev = ASR_csv_DataSet(
        list_files=args.dirs.dev.list_files,
        args=args,
        _shuffle=False,
        transform=False)
    dataset_test = ASR_csv_DataSet(
        list_files=args.dirs.test.list_files,
        args=args,
        _shuffle=False,
        transform=True)
elif args.dirs.type == 'scp_classify':
    from .dataset import ASR_classify_ArkDataSet
    dataset_train = ASR_classify_ArkDataSet(
        scp_file=args.dirs.train.scp,
        class_file=args.dirs.train.label,
        args=args,
        _shuffle=True)
    dataset_dev = ASR_classify_ArkDataSet(
        scp_file=args.dirs.dev.scp,
        class_file=args.dirs.dev.label,
        args=args,
        _shuffle=False)
    dataset_test = ASR_classify_ArkDataSet(
        scp_file=args.dirs.test.scp,
        class_file=args.dirs.test.label,
        args=args,
        _shuffle=False)
elif args.dirs.type == 'test':
    dataset_dev = dataset_train = dataset_test = None
else:
    raise NotImplementedError('not dataset type!')
args.dataset_dev = dataset_dev
args.dataset_train = dataset_train
args.dataset_test = dataset_test

try:
    args.data.dim_feature = TFData.read_tfdata_info(args.dirs.train.tfdata)['dim_feature']
    args.data.train_size = TFData.read_tfdata_info(args.dirs.train.tfdata)['size_dataset']
    args.data.dev_size = TFData.read_tfdata_info(args.dirs.dev.tfdata)['size_dataset']
    args.data.dim_input = args.data.dim_feature * \
            (args.data.right_context + args.data.left_context +1) *\
            (2 if args.data.add_delta else 1)
except:
    print("have not converted to tfdata yet: ")

# model
## encoder
if args.model.encoder.type == 'transformer_encoder':
    from models.encoders.transformer_encoder import Transformer_Encoder as encoder
elif args.model.encoder.type == 'conv_lstm':
    from models.encoders.conv_lstm import CONV_LSTM as encoder
elif args.model.encoder.type == 'classifier':
    from models.encoders.classifier import CONV_LSTM_Classifier as encoder
elif args.model.encoder.type == 'BLSTM':
    from models.encoders.blstm import BLSTM as encoder
elif args.model.encoder.type == 'conv':
    from models.encoders.conv import CONV as encoder
elif args.model.encoder.type == 'conv2':
    from models.encoders.conv2 import CONV as encoder
else:
    raise NotImplementedError('not found encoder type: {}'.format(args.model.encoder.type))
args.model.encoder.type = encoder

## decoder
# try:
if args.model.decoder.type == 'FC':
    from models.decoders.fc_decoder import FCDecoder as decoder
elif args.model.decoder.type == 'classifier':
    from models.decoders.classifier import FCDecoder as decoder
elif args.model.decoder.type == 'transformer_decoder':
    from models.decoders.transformer_decoder import Transformer_Decoder as decoder
else:
    raise NotImplementedError('not found decoder type: {}'.format(args.model.decoder.type))
args.model.decoder.type = decoder
# except:
#     print("not using decoder!")
#     args.model.decoder = AttrDict()
#     args.model.decoder.size_embedding = None
#     args.model.decoder.type = None

## model
if args.model.type == 'Seq2SeqModel':
    from models.seq2seqModel import Seq2SeqModel as Model
elif args.model.type == 'ctcModel':
    from models.ctcModel import CTCModel as Model
elif args.model.type == 'ctcModel_EODM':
    from models.ctcModel_EODM import CTCModel as Model
elif args.model.type == 'classifier':
    from models.classifier import Classifier as Model
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

    from models.gan import GAN
    args.GAN = GAN
