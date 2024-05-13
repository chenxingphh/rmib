from .interface import Interface
from transformers import AutoTokenizer
import random


class InterfaceSBert(Interface):

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

        s1 = self.tokenizer(text1,
                            max_length=int(self.args.max_len) // 2,
                            truncation=True,
                            padding='max_length',
                            # return_tensors='pt'
                            )

        s2 = self.tokenizer(text2,
                            max_length=int(self.args.max_len) // 2,
                            truncation=True,
                            padding='max_length',
                            # return_tensors='pt'
                            )

        processed = {
            'text1': s1['input_ids'],
            'text1_token_type_ids': s1['token_type_ids'],
            'text1_attention_mask': s1['attention_mask'],
            'text2': s2['input_ids'],
            'text2_token_type_ids': s2['token_type_ids'],
            'text2_attention_mask': s2['attention_mask'],
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
