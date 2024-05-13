from .interface import Interface
from transformers import AutoTokenizer
import random
import numpy as np


class InterfaceBert(Interface):

    def __init__(self, args, log=None):
        super().__init__(args, log=None)

        model_path = args.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def pre_process(self, data, training=True):
        result = [self.process_sample(sample) for sample in data]

        batch_size = self.args.batch_size
        return [result[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def process_sample(self, sample, with_target=True):
        text1 = sample['text1']
        text2 = sample['text2']

        if self.args.lower_case:
            text1 = text1.lower()
            text2 = text2.lower()

        s = self.tokenizer(text=text1,
                           text_pair=text2,
                           max_length=self.args.max_len,
                           truncation=True,
                           padding='max_length',
                           )

        text1_mask = np.asarray(s['attention_mask']) - np.asarray(s['token_type_ids'])
        text1_mask = text1_mask.tolist()
        text2_mask = s['token_type_ids']

        processed = {
            'text': s['input_ids'],
            'text_token_type_ids': s['token_type_ids'],
            'text_attention_mask': s['attention_mask'],
            'text1_mask': text1_mask,
            'text2_mask': text2_mask,
        }

        if 'target' in sample and with_target:
            target = sample['target']
            assert target in self.target_map
            processed['target'] = self.target_map.index(target)

        return processed

    def shuffle_batch(self, data):
        data = random.sample(data, len(data))
        return list(map(self.make_batch, data))

    def make_batch(self, batch, with_target=True):
        batch = {key: [sample[key] for sample in batch] for key in batch[0].keys()}
        if 'target' in batch and not with_target:
            del batch['target']

        return batch
