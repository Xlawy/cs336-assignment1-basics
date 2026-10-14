import regex as re


def find_merge_pair(counts_pair):
    def ranking(pair):
        return (counts_pair[pair], pair)

    return max(counts_pair, key=ranking)


def find_counts_pair(counts_bytes):
    counts_pair = {}
    for fragment, cnt in counts_bytes.items():
        for i in range(len(fragment) - 1):
            pair = (fragment[i], fragment[i + 1])
            if pair in counts_pair:
                counts_pair[pair] += cnt
            else:
                counts_pair[pair] = cnt

    return counts_pair


def merge(counts_pair, counts_bytes, pair):
    merge_bytes = {}
    for fragment, cnt in counts_bytes.items():
        res = []
        i = 0
        if pair[0] not in fragment:
            merge_bytes[fragment] = cnt
            continue
        while i < len(fragment):
            if (i + 1) < len(fragment) and (fragment[i], fragment[i + 1]) == pair:
                res.append(fragment[i] + fragment[i + 1])
                i += 2
            else:
                res.append(fragment[i])
                i += 1

        new_fragment = tuple(res)
        if new_fragment != fragment:
            for i in range(len(fragment) - 1):
                old_pair = (fragment[i], fragment[i + 1])
                counts_pair[old_pair] -= cnt

                if counts_pair[old_pair] == 0:
                    del counts_pair[old_pair]

            for i in range(len(new_fragment) - 1):
                new_pair = (new_fragment[i], new_fragment[i + 1])
                if new_pair in counts_pair:
                    counts_pair[new_pair] += cnt
                else:
                    counts_pair[new_pair] = cnt

        merge_bytes[new_fragment] = cnt
    return merge_bytes


def train(text_path, vocab_size, special_token):
    counts = {}
    counts_bytes = {}
    merges = []
    vocab = {}
    INIT_VOCAB_SIZE = 256
    special_token_size = len(special_token)
    least_vocab_size = INIT_VOCAB_SIZE + special_token_size
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    with open(text_path, "r", encoding="utf-8") as file:
        text = file.read()

    if vocab_size < least_vocab_size:
        raise ValueError("vocab_size must be at least ", least_vocab_size)

    for idx in range(INIT_VOCAB_SIZE):
        vocab[idx] = bytes([idx])

    for idx in range(INIT_VOCAB_SIZE, least_vocab_size):
        s_token_id = idx - INIT_VOCAB_SIZE
        s_token = special_token[s_token_id]
        s_token_utf8 = s_token.encode("utf-8")
        vocab[idx] = s_token_utf8

    # re.escape 起到转义作用 比如正则的.表示任意字符 用这个可以把.转义,用于匹配.,转成\.
    # join的作用是把多个连接起来,最开始的"|"是分隔符
    parts_separators_pat = "|".join(re.escape(s) for s in special_token)
    if parts_separators_pat != "":
        parts = re.split(parts_separators_pat, text)
    else:
        parts = [text]

    for part in parts:
        for match in re.finditer(PAT, part):
            fragment = match.group(0)
            if fragment in counts:
                counts[fragment] += 1
            else:
                counts[fragment] = 1

    for fragment, cnt in counts.items():
        frag_utf8 = fragment.encode("utf-8")
        res = []
        for value in frag_utf8:
            byte = bytes([value])
            res.append(byte)
        res_tuple = tuple(res)
        counts_bytes[res_tuple] = cnt

    counts_pair = find_counts_pair(counts_bytes)
    for _ in range(vocab_size - least_vocab_size):
        if not counts_pair:
            break

        merged_pair = find_merge_pair(counts_pair)
        merges.append(merged_pair)
        counts_bytes = merge(counts_pair, counts_bytes, merged_pair)

    for pair in merges:
        vocab[len(vocab)] = pair[0] + pair[1]

    return vocab, merges


def main():
    text_path = "data/text.txt"
    special_token = ["<|endoftext|>"]
    vocab_size = 30000
    vocab, merges = train(text_path, vocab_size, special_token)
    print(len(vocab))
    print(merges)


if __name__ == "__main__":
    main()
