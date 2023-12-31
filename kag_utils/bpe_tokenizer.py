from tqdm.auto import tqdm
from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

class BPETokenizer:
    ST = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tok.normalizer = normalizers.Sequence([normalizers.NFC()])
        self.tok.pre_tokenizer = pre_tokenizers.ByteLevel()
        self.tok.post_processor = processors.TemplateProcessing(
            single="[CLS] $A",
            special_tokens=[("[CLS]", 1)],
        )
        
    @classmethod
    def chunk_dataset(cls, dataset, chunk_size=1_000):
        for i in range(0, len(dataset), chunk_size):
            yield dataset[i : i + chunk_size]["text"]
        
    def train(self, data):
        trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.ST)
        dataset = Dataset.from_pandas(data[["text"]])
        self.tok.train_from_iterator(self.chunk_dataset(dataset), trainer=trainer)
        return self
    
    def tokenize(self, data):
        tokenized_texts = []
        for text in tqdm(data['text'].tolist()):
            tokenized_texts.append(self.tok.encode(text))
        return tokenized_texts
    
    def get_fast_tokenizer(self, max_length):
        return PreTrainedTokenizerFast(
            tokenizer_object=self.tok,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            model_max_length=max_length
        )
