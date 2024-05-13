import os
from shutil import copyfile


def copy(src, tgt):
    copyfile(os.path.abspath(src), os.path.abspath(tgt))


os.makedirs('wikiqa', exist_ok=True)

copy('orig/WikiQACorpus/WikiQA-dev-filtered.ref', 'wikiqa/dev.ref')
copy('orig/WikiQACorpus/WikiQA-test-filtered.ref', 'wikiqa/test.ref')
copy('orig/WikiQACorpus/emnlp-table/WikiQA.CNN.dev.rank', 'wikiqa/dev.rank')
copy('orig/WikiQACorpus/emnlp-table/WikiQA.CNN.test.rank', 'wikiqa/test.rank')
for split in ['train', 'dev', 'test']:
    print('processing WikiQA', split)
    copy('orig/WikiQACorpus/WikiQA-{}.txt'.format(split), 'wikiqa/{}.txt'.format(split))
