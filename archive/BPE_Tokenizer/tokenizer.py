import regex as re
import json


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        if not special_tokens:
            self.special_tokens = []
        else:
            # 避免共同前缀的special_token先匹配短的，优先匹配长的
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)

        self.vocab_reverse = {}
        for id, token in vocab.items():
            self.vocab_reverse[token] = id

    # 加上这个才是类方法
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r", encoding="utf-8") as file:
            vocab_json = json.load(file)

        with open(merges_filepath, "r", encoding="utf-8") as file:
            merges_json = json.load(file)

        vocab = {}
        merges = []

        for idx, bytes_list in vocab_json.items():
            vocab[int(idx)] = bytes(bytes_list)

        for merge_json in merges_json:
            part1, part2 = merge_json
            merges.append((bytes(part1), bytes(part2)))

        return cls(vocab, merges, special_tokens)

    def encode(self, text):
        # 预期输入，字符串
        # 预期输出，tokens列表
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        tokens = []
        text_bytes = []

        if not self.special_tokens:
            text_split = [text]
        else:
            parts_separators_pat = "|".join(re.escape(s) for s in self.special_tokens)
            parts_separators_pat = "(" + parts_separators_pat + ")"
            text_split = re.split(parts_separators_pat, text)

        for piece in text_split:
            if piece in self.special_tokens:
                tokens.append(self.vocab_reverse[piece.encode("utf-8")])
                continue
            for match in re.finditer(PAT, piece):
                text_bytes = []
                for piece in match.group(0).encode("utf-8"):
                    text_bytes.append(bytes([piece]))

                for pair in self.merges:
                    merged = []
                    i = 0

                    while i < len(text_bytes):
                        if i + 1 < len(text_bytes) and (text_bytes[i], text_bytes[i + 1]) == pair:
                            merged.append(text_bytes[i] + text_bytes[i + 1])
                            i += 2
                        else:
                            merged.append(text_bytes[i])
                            i += 1

                    text_bytes = merged

                for text_byte in text_bytes:
                    tokens.append(self.vocab_reverse[text_byte])

        return tokens

    def encode_iterable(self, iterable):
        for text in iterable:
            tokens = self.encode(text)
            yield from tokens

    def decode(self, tokens):
        # 预期输入，tokens列表
        # 预期输出，字符串
        res = ""
        res_bytes_list = []

        for token in tokens:
            res_bytes_list.append(self.vocab[token])

        # 这里不用一个循环来 +=，用join可以直接连接
        # 如果用 res_bytes += bytes，会效率低下，因为bytes本身不可变，通过 += 的本质是创建了一个更大的bytes对象，然后逐个复制进去，会让效率低下
        # join连接list的时候是一次性计算所有长度，然后一次性分配一个足够大的空间来存储
        res_bytes = b"".join(res_bytes_list)
        # 加上errors="replace"，遇到不合法的子集序列会用特殊字符进行替代
        res = res_bytes.decode("utf-8", errors="replace")
        return res


def main():
    vocab = {
        0: b" ",
        1: b"a",
        2: b"b",
        3: b"c",
        4: b"ab",
        5: b"bc",
        6: b"abc",
        7: b"\xe4",
        8: b"\xb8",
        9: b"\xad",  # 7 8 9 -> 中
        10: b"<|endoftext|>",
        11: b"<|endoftext|><|endoftext|>",
    }
    merges = [(b"a", b"b"), (b"b", b"c"), (b"ab", b"c")]
    special_tokens = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]

    # tokenizer = Tokenizer(vocab, merges, special_tokens)

    # text_input = "abc<|endoftext|><|endoftext|>abc<|endoftext|>"
    # text_encode = tokenizer.encode(text_input)
    # print(text_encode)

    # tokens_input = []
    # tokens_decode = tokenizer.decode(tokens_input)
    # print(tokens_decode)

    tokenizer = Tokenizer.from_files("data/text_vocabs.json", "data/text_merges.json")
    print(tokenizer.vocab)
    print(tokenizer.merges)


if __name__ == "__main__":
    main()
