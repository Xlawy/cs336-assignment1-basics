import regex as re


def find_merge_pair(counts_pair):
    def ranking(pair):
        return (counts_pair[pair], pair)

    return max(counts_pair, key=ranking)


def counts_insert(counts, key, value):
    if key in counts:
        counts[key] += value
    else:
        counts[key] = value


def merge(counts_bytes, counts_pair, merged_pair):
    new_counts_bytes = {}
    for text_bytes, cnt in counts_bytes.items():
        # 更新counts_bytes
        if merged_pair[0] not in text_bytes or merged_pair[1] not in text_bytes:
            new_counts_bytes[text_bytes] = cnt
            continue
        idx = 0
        new_text_bytes = []
        # 逐一匹配该text是否包括merged_pair
        while idx < len(text_bytes):
            if idx < len(text_bytes) - 1 and (text_bytes[idx], text_bytes[idx + 1]) == merged_pair:
                new_text_bytes.append(text_bytes[idx] + text_bytes[idx + 1])
                idx += 2
            else:
                new_text_bytes.append(text_bytes[idx])
                idx += 1
        new_counts_bytes[tuple(new_text_bytes)] = cnt

        # 更新counts_pair
        # 遍历有过修改的text_bytes，删除所有旧pair，增加所有新pair
        if tuple(new_text_bytes) != text_bytes:
            # 删除所有旧pair
            for i in range(len(text_bytes) - 1):
                old_pair = (text_bytes[i], text_bytes[i + 1])
                counts_pair[old_pair] -= cnt

                if counts_pair[old_pair] == 0:
                    del counts_pair[old_pair]

            # 增加所有新pair
            for i in range(len(new_text_bytes) - 1):
                new_pair = (new_text_bytes[i], new_text_bytes[i + 1])
                counts_insert(counts_pair, new_pair, cnt)
    return new_counts_bytes, counts_pair


def init_vocab(vocab_size, special_tokens):
    vocabs = {}
    # 覆盖256字节和special_token
    init_vocab_size = 256 + len(special_tokens)
    if vocab_size < init_vocab_size:
        raise ValueError("vocab_size must be at least: ", init_vocab_size)

    # 初始化256字节的vocabs
    for idx in range(256):
        vocabs[idx] = bytes([idx])

    # 初始化special_tokens
    for idx in range(256, init_vocab_size):
        vocabs[idx] = special_tokens[idx - 256].encode("utf-8")

    return vocabs


def init_text(text_path, special_tokens):
    with open(text_path, "r", encoding="utf-8") as file:
        text = file.read()

    # 使用special_tokens对原始text进行分割
    # regex.escape用于处理字符串中的正则字符，加上转义，防止被正则处理，"." -> "\."
    # join方法会用前面的字符串当连接词对后面的字符串进行连接
    # 防止出现相同前缀的specialtoken 排序一下 长的放前面
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    print(special_tokens)
    parts_separator = "|".join(re.escape(s) for s in special_tokens)
    if parts_separator != "":
        text_parts = re.split(parts_separator, text)
    else:
        text_parts = [text]

    return text_parts


def train(text_path, special_tokens, vocab_size):
    merges = []
    # 初始化vocabs
    vocabs = init_vocab(vocab_size, special_tokens)

    # 读取文本，并使用special_tokens进行分割
    text_parts = init_text(text_path, special_tokens)

    # 使用正则分割，统计str片段出现次数
    # 也就是pre-tokenization
    counts_str = {}
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for text_part in text_parts:
        for text_str_match in re.finditer(PAT, text_part):
            counts_insert(counts_str, text_str_match.group(0), 1)

    # 把str转成单字节bytes
    counts_bytes = {}
    for text_str, cnt in counts_str.items():
        text_utf8 = text_str.encode("utf-8")
        text_bytes = []
        for text_char_bytes in text_utf8:
            # bytes()接收的是一个int数值，如果bytes(100)，就是创建100个零字节
            # 如果接收的是一个list，就是根据list创建字节序列，bytes([97, 98]) -> b"ab"
            text_bytes.append(bytes([text_char_bytes]))
        text_bytes_tuple = tuple(text_bytes)
        counts_bytes[text_bytes_tuple] = cnt

    # pairs计数
    counts_pair = {}
    for text_bytes_tuple, cnt in counts_bytes.items():
        for i in range(len(text_bytes_tuple) - 1):
            pair = (text_bytes_tuple[i], text_bytes_tuple[i + 1])
            counts_insert(counts_pair, pair, cnt)

    # 训练轮数
    for _ in range(vocab_size - 256 - len(special_tokens)):
        # 循环重点 break
        if not counts_pair:
            break

        # 寻找出现次数最多的pair
        merged_pair = find_merge_pair(counts_pair)
        merges.append(merged_pair)
        # 合并pair, 更新counts_bytes和counts_pair
        counts_bytes, counts_pair = merge(counts_bytes, counts_pair, merged_pair)

    for pair in merges:
        vocabs[len(vocabs)] = pair[0] + pair[1]

    return vocabs, merges


def main():
    text = "nihao nihao nihao ni ni hao<|endoftext|>nihao hello he llo"
    special_tokens = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
    vocab_size = 300

    vocabs, merges = train("data/bpe_train_test.txt", special_tokens, vocab_size)

    print("vocabs: ", vocabs)
    print("merges: ", merges)


if __name__ == "__main__":
    main()
