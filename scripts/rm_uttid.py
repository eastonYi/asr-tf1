import logging
from argparse import ArgumentParser
from collections import Counter, defaultdict
from tqdm import tqdm


def main(f_in, f_out):
    with open(f_in) as f, open(f_out, 'w') as fw:
        for line in f:
            uttid, chars = line.strip().split(' ', maxsplit=1)
            fw.write(chars + '\n')


if __name__ == '__main__':
    """
    f_in = '/home/easton/mount/114/data/CALLHOME/Mandarin/test_dim80/test.char'
    f_out = '/home/easton/mount/114/data/CALLHOME/Mandarin/test_dim80/test.phone'
    """
    parser = ArgumentParser()
    parser.add_argument('--output', type=str, dest='f_out')
    parser.add_argument('--input', type=str, dest='f_in')
    args = parser.parse_args()
    # Read config
    logging.basicConfig(level=logging.INFO)
    main(args.f_in, args.f_out)
    logging.info("Done")
