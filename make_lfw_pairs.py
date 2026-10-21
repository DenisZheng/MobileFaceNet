"""Convert the official LFW pairs.txt into foamliu's lfw_test_pair.txt.

lfw_eval.evaluate() expects data/lfw_test_pair.txt with 6,000 lines of

    Name/Name_0001.jpg Name/Name_0004.jpg 1      (same person)
    NameA/NameA_0001.jpg NameB/NameB_0002.jpg 0  (different)

but upstream never shipped a generator for it (the contributor's
create_lfw_test_pair.py expects pre-split pairs_XX.txt files in yet
another format). The official pairs.txt -- downloaded together with
lfw-funneled.tgz from http://vis-www.cs.umass.edu/lfw/ -- has a header
line ("10 300") followed by 3-token (same) and 4-token (different)
lines with bare image numbers. This script converts it.

Usage:
    python make_lfw_pairs.py                 # data/pairs.txt -> data/lfw_test_pair.txt
    python make_lfw_pairs.py --src /path/to/pairs.txt
"""
import argparse
import os


def img_relpath(name, num):
    return '{}/{}_{:04d}.jpg'.format(name, name, int(num))


def convert(src, dst):
    with open(src, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # Drop the header line ("10\t300") if present: it has 2 numeric tokens.
    first = lines[0].replace('\t', ' ').split()
    if len(first) == 2 and all(t.isdigit() for t in first):
        lines = lines[1:]

    out = []
    for ln in lines:
        tok = ln.replace('\t', ' ').split()
        if len(tok) == 3:        # same person: name, img1, img2
            name, a, b = tok
            out.append('{} {} 1'.format(img_relpath(name, a), img_relpath(name, b)))
        elif len(tok) == 4:      # different: name1, img1, name2, img2
            n1, a, n2, b = tok
            out.append('{} {} 0'.format(img_relpath(n1, a), img_relpath(n2, b)))
        else:
            raise ValueError('unexpected pairs.txt line: {!r}'.format(ln))

    with open(dst, 'w') as f:
        f.write('\n'.join(out) + '\n')
    print('wrote {} pairs to {}'.format(len(out), dst))
    if len(out) != 6000:
        print('WARNING: expected 6000 pairs for the standard LFW protocol')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert official LFW pairs.txt to lfw_test_pair.txt')
    parser.add_argument('--src', default='data/pairs.txt')
    parser.add_argument('--dst', default='data/lfw_test_pair.txt')
    args = parser.parse_args()
    if not os.path.isfile(args.src):
        raise SystemExit('{} not found -- download pairs.txt from '
                         'http://vis-www.cs.umass.edu/lfw/pairs.txt'.format(args.src))
    convert(args.src, args.dst)
