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

        # 构建反向vocab 用于encode
        self.vocab_reverse = {}
        for token, bytes_seq in vocab.items():
            self.vocab_reverse[bytes_seq] = token

        # 应对新增special_token的情况
        if special_tokens:
            for special_token in special_tokens:
                special_token_utf8 = special_token.encode("utf-8")
                if special_token_utf8 not in self.vocab_reverse:
                    idx_max = max(vocab)
                    self.vocab[idx_max + 1] = special_token_utf8
                    self.vocab_reverse[special_token_utf8] = idx_max + 1

    # 加上这个@才是类方法
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r", encoding="utf-8") as file:
            vocabs_json = json.load(file)

        with open(merges_filepath, "r", encoding="utf-8") as file:
            merges_json = json.load(file)

        vocabs = {}
        merges = []

        for idx, bytes_list in vocabs_json.items():
            vocabs[int(idx)] = bytes(bytes_list)

        for merge_json in merges_json:
            part1, part2 = merge_json
            merges.append((bytes(part1), bytes(part2)))

        return cls(vocabs, merges, special_tokens)

    def encode(self, text):
        # 预期输入，字符串
        # 预期输出，tokens列表
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        tokens = []

        # 通过special_token把text分割成多个parts
        if not self.special_tokens:
            text_parts = [text]
        else:
            text_parts_separator = "|".join(re.escape(s) for s in self.special_tokens)
            # 这里在两边加上()是为了形成捕获组
            # 这样通过split得到的切分结果是包括了分隔符的
            text_parts_separator = "(" + text_parts_separator + ")"
            text_parts = re.split(text_parts_separator, text)

        # 对每个part依次处理
        # 步骤和训练其实比较接近,都是先用PAT分割,然后bytes字节序列化
        # 用merges进行合并,最后转成tokens
        for text_part in text_parts:
            # 如果是special_token,直接去vocab里面找token id
            if text_part in self.special_tokens:
                tokens.append(self.vocab_reverse[text_part.encode("utf-8")])
                continue

            # 对正则匹配到的text逐个处理
            for match in re.finditer(PAT, text_part):
                text_utf8 = match.group(0).encode("utf-8")
                text_bytes = []
                # 转化成bytes序列
                for text_char_bytes in text_utf8:
                    text_bytes.append(bytes([text_char_bytes]))

                # 通过merges进行合并
                for pair in self.merges:
                    merged_text = []
                    if pair[0] not in text_bytes or pair[1] not in text_bytes:
                        continue

                    i = 0
                    while i < len(text_bytes):
                        if (i + 1) < len(text_bytes) and (text_bytes[i], text_bytes[i + 1]) == pair:
                            merged_text.append(text_bytes[i] + text_bytes[i + 1])
                            i += 2
                        else:
                            merged_text.append(text_bytes[i])
                            i += 1

                    text_bytes = merged_text

                for text_char_bytes in text_bytes:
                    tokens.append(self.vocab_reverse[text_char_bytes])

        return tokens

    def encode_iterable(self, iterable):
        for text in iterable:
            tokens = self.encode(text)
            yield from tokens

    def decode(self, tokens):
        # 预期输入 tokens序列
        # 预期输出 字符串str
        res = ""
        res_bytes = b""
        res_bytes_list = []

        for token in tokens:
            res_bytes_list.append(self.vocab[token])

        # 这里不用一个循环来 +=，用join可以直接连接
        # 如果用 res_bytes += bytes，会效率低下，因为bytes本身不可变，通过 += 的本质是创建了一个更大的bytes对象，然后逐个复制进去，会让效率低下
        # join连接list的时候是一次性计算所有长度，然后一次性分配一个足够大的空间来存储
        res_bytes = res_bytes.join(res_bytes_list)
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
    special_tokens = ["<|endoftext|>", "<|endoftext|><|endoftext|>", "<|zjh|>"]

    tokenizer = Tokenizer(vocab, merges, special_tokens)

    text_input = "abc<|endoftext|><|endoftext|>abc<|endoftext|>"
    text_encode = tokenizer.encode(text_input)
    print(text_encode)

    # tokens_input = [2, 7, 8, 9]
    # tokens_decode = tokenizer.decode(tokens_input)
    # print(tokens_decode)

    # tokenizer = Tokenizer.from_files("data/text_vocabs.json", "data/text_merges.json")
    # print(tokenizer.vocab)
    # print(tokenizer.merges)


if __name__ == "__main__":
    main()
