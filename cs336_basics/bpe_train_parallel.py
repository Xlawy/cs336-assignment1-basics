import regex as re
from typing import BinaryIO
from multiprocessing import Pool
import os
import json


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


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenization_serial(text, special_tokens):
    # 使用special_tokens对原始text进行分割
    # regex.escape用于处理字符串中的正则字符，加上转义，防止被正则处理，"." -> "\."
    # join方法会用前面的字符串当连接词对后面的字符串进行连接
    # 防止出现相同前缀的specialtoken 排序一下 长的放前面
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    parts_separator = "|".join(re.escape(s) for s in special_tokens)
    if parts_separator != "":
        text_parts = re.split(parts_separator, text)
    else:
        text_parts = [text]

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

    return counts_bytes


def pre_tokenization_part_file(text_path, start, end, special_tokens):
    with open(text_path, "rb") as file:
        file.seek(start)
        text_bytes = file.read(end - start)

    text = text_bytes.decode("utf-8")
    return pre_tokenization_serial(text, special_tokens)


def pre_tokenization_parallel(text_path, special_tokens, num_processors):
    # 读取文件
    with open(text_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_processors, b"<|endoftext|>")

    # 分配任务
    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        task = (text_path, start, end, special_tokens)
        tasks.append(task)

    counts = []
    if not tasks:
        counts = []
    elif num_processors == 1:
        for task in tasks:
            count = pre_tokenization_part_file(*task)
            counts.append(count)
    else:
        worker_count = min(num_processors, len(tasks))

        with Pool(processes=worker_count) as pool:
            counts = pool.starmap(pre_tokenization_part_file, tasks)

    counts_bytes = {}
    for count in counts:
        for text_bytes, cnt in count.items():
            counts_insert(counts_bytes, text_bytes, cnt)

    return counts_bytes


def train(text_path, special_tokens, vocab_size, num_processors):
    merges = []
    # 初始化vocabs
    vocabs = init_vocab(vocab_size, special_tokens)

    # 使用special_token和正则进行分割，统计str片段出现次数
    # 并行pre-tokenization
    counts_bytes = pre_tokenization_parallel(text_path, special_tokens, num_processors)

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


def export_json(vocabs, merges):
    vocabs_json = {}
    merges_json = []

    # bytes不能直接导出到json
    # 所以使用list转成int序列
    for idx, string_bytes in vocabs.items():
        vocabs_json[idx] = list(string_bytes)

    for pair in merges:
        part1, part2 = pair
        merge_json = []
        merge_json.append(list(part1))
        merge_json.append(list(part2))
        merges_json.append(merge_json)

    with open("data/text_vocabs.json", "w", encoding="utf-8") as file:
        json.dump(vocabs_json, file, indent=2)

    with open("data/text_merges.json", "w", encoding="utf-8") as file:
        json.dump(merges_json, file, indent=2)


def main():
    text_path = "data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
    vocab_size = 300
    num_processors = 16

    vocabs, merges = train(text_path, special_tokens, vocab_size, num_processors)

    export_json(vocabs, merges)

    # print("vocabs: ", vocabs)
    # print("merges: ", merges)


if __name__ == "__main__":
    main()
