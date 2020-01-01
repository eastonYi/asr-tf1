import logging
from argparse import ArgumentParser
from collections import Counter, defaultdict
from tqdm import tqdm
import re
from pypinyin import pinyin, Style


def main(f_in, f_out):
    with open(f_in) as f, open(f_out, 'w') as fw:
        for line in f:
            uttid, chars = line.strip().split(' ', maxsplit=1)
            list_phones = []
            for p in pinyin(chars.replace(' ', ''), style=Style.TONE3):
                p = p[0]
                tone = re.findall('[0-9]', p) # [['3'], []]
                tone = tone[0] if tone else '5'
                tone = ''
                try:
                    p = re.findall('[a-z]+', p)[0]
                except:
                    import pdb; pdb.set_trace()
                    print(p, uttid)
                phones = pinyin2phones(p) + tone
                list_phones.append(phones)
            fw.write(uttid + ' ' +' '.join(list_phones) + '\n')

def pinyin2phones(p):
    dict_case = {'wu':'uu u', 'wen':'uu un', 'wei':'uu ui', 'weng':'uu ueng',
                 'yu':'vv v', 'yue':'vv ve', 'yuan':'vv van', 'yun':'vv vn',
                 'you':'ii iu','yang':'ii iang', 'ye':'ii ie', 'yong':'ii iong',
                 'ri':'r iz', 'zi':'z iy', 'si':'s iy', 'ci':'c iy', 'n':'ee en'}
    if p in dict_case.keys():
        phones = dict_case[p]
    elif p.startswith('a'):
        phones = 'aa a'+ p[1:]
    elif p.startswith('e'):
        phones = 'ee e'+ p[1:]
    elif p.startswith('o'):
        phones = 'oo o'+ p[1:]
    elif p.startswith('w'):
        phones = 'uu u'+ p[1:]
    elif p.startswith('qu'):
        phones = 'q v'+ p[2:]
    elif p.startswith('xu'):
        phones = 'x v'+ p[2:]
    elif p.startswith('qu'):
        phones = 'q v'+ p[2:]
    elif p.startswith('ju'):
        phones = 'j v'+ p[2:]
    elif p.startswith('ya'):
        phones = 'ii ia'+ p[2:]
    elif p.endswith('hi'):
        phones = p[:-2] + 'h ix'
    elif p[0] == 'w':
        phones = 'uu ' + p[1:]
    elif p[0] == 'y':
        phones = 'ii ' + p[1:]
    elif 'yu' in p:
        phones = 'vv v' + p[2:]
    elif 'ue' in p:
        phones = p[:-2] + ' ve'
    else:
        if p[1] == 'h':
            phones = p[:2] + ' ' + p[2:]
        else:
            phones = p[0] + ' ' + p[1:]

    return phones

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
