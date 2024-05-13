import os
from tqdm import tqdm

print('processing quora')
os.makedirs('quora', exist_ok=True)

for split in ('train', 'dev', 'test'):
    with open('orig/Quora_question_pair_partition/{}.tsv'.format(split)) as f, \
            open('quora/{}.txt'.format(split), 'w') as fout:
        n_lines = 0
        for _ in f:
            n_lines += 1
        f.seek(0)
        for line in tqdm(f, total=n_lines, leave=False):
            elements = line.rstrip().split('\t')
            fout.write('{}\t{}\t{}\n'.format(elements[1], elements[2], int(elements[0])))
