from collections import Counter


class BPE:

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def fit(self, text: str):
        self.id2token = dict()
        self.token2id = dict()
        unique_tokens = sorted(list(set(text)))
        text_list = list(text)

        while len(unique_tokens) < self.vocab_size:
            pairs = []
            for i in range(1, len(text_list)):
                pairs.append(text_list[i - 1] + text_list[i])

            if not pairs:
                break

            pair_counter = Counter(pairs)
            freq_pair, max_freq = pair_counter.most_common(1)[0]

            # Если самая частая пара встречается только 1 раз, дальше не имеет смысла
            if max_freq <= 1:
                break

            unique_tokens.append(freq_pair)

            # Объединяем пары в тексте
            new_text = []
            i = 0
            while i < len(text_list):
                if (
                    i < len(text_list) - 1
                    and text_list[i] + text_list[i + 1] == freq_pair
                ):
                    new_text.append(freq_pair)
                    i += 2
                else:
                    new_text.append(text_list[i])
                    i += 1

            text_list = new_text

        for i, val in enumerate(unique_tokens):
            self.id2token[i] = val
            self.token2id[val] = i

    def encode(self, text: str) -> list:
        res = []
        while text:
            tokens = []
            for token in self.token2id.keys():
                if text[0] == token[0]:
                    tokens.append(token)
            for token in sorted(tokens, key=len, reverse=True):
                if text.find(token) == 0:
                    res.append(self.token2id[token])
                    text = text[len(token) :]
                    break
        return res