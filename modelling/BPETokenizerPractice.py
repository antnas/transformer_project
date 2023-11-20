from transformers import AutoTokenizer
from collections import defaultdict

class BPETokenizer():
    def __init__(self, vocab_size=64):
        self.vocab_size = vocab_size
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.word_freqs = defaultdict(int)
        self.vocab = []
        self.merges = []

    def train(self, corpus):
        # First we calculate the word frequencies of the corpus
        for text in corpus:
            words_with_offsets = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                self.word_freqs[word] += 1

        # Determine the splits for each word
        splits = {word: [c for c in word] for word in self.word_freqs.keys()}

        # Determine the alphabet used in the corpus
        alphabet = []
        for word in self.word_freqs.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)
        alphabet.sort()
        self.vocab = vocab = ["<|endoftext|>"] + alphabet.copy()

        pair_freqs = self.compute_pair_freqs(splits)

        for i, key in enumerate(pair_freqs.keys()):
            print(f"{key}: {pair_freqs[key]}")
            if i >= 5:
                break

        best_pair = ""
        max_freq = None

        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq

        while len(vocab) < self.vocab_size:
            pair_freqs = self.compute_pair_freqs(splits)
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            splits = merge_pair(*best_pair, splits)
            merges[best_pair] = best_pair[0] + best_pair[1]
            vocab.append(best_pair[0] + best_pair[1])

    def compute_pair_freqs(self, splits):
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def merge_pair(self, a, b, splits):
        for word in self.word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2:]
                else:
                    i += 1
            splits[word] = split
        return splits

    def tokenize(self, text):
        pre_tokenize_result = self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        splits = [[l for l in word] for word in pre_tokenized_text]
        for pair, merge in merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits[idx] = split

        return sum(splits, [])

    def train(self, corpus, target_vocab_size):
        while len(self.vocab) < target_vocab_size:
            pair_freqs = self.compute_pair_freqs(splits)
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            splits = merge_pair(*best_pair, splits)
            merges[best_pair] = best_pair[0] + best_pair[1]
            vocab.append(best_pair[0] + best_pair[1])

corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

merges = {("Ġ", "t"): "Ġt"}
vocab.append("Ġt")

splits = merge_pair("Ġ", "t", splits)
print(splits)