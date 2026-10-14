import regex as re
import os
from typing import BinaryIO
from multiprocessing import Pool
import time


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


def pre_train(text, special_token):
    counts = {}
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # re.escape 起到转义作用 比如正则的.表示任意字符 用这个可以把.转义,用于匹配.,转成\.
    # join的作用是把多个连接起来,最开始的"|"是分隔符
    parts_separators_pat = "|".join(re.escape(s) for s in special_token)
    if parts_separators_pat != "":
        parts = re.split(parts_separators_pat, text)
    else:
        parts = [text]

    for idx, part in enumerate(parts):
        # print("[Pre-Training] it", idx + 1, "/ total ", len(parts))
        for match in re.finditer(PAT, part):
            fragment = match.group(0)
            if fragment in counts:
                counts[fragment] += 1
            else:
                counts[fragment] = 1

    return counts


def count_chunk(text_path, start, end, special_token):
    with open(text_path, "rb") as file:
        file.seek(start)
        chunk_bytes = file.read(end - start)

    text = chunk_bytes.decode("utf-8")

    return pre_train(text, special_token)


def train(text_path, vocab_size, special_token, num_precesses):
    counts_bytes = {}
    chunk_counts = []
    total_counts = {}
    merges = []
    vocab = {}
    INIT_VOCAB_SIZE = 256
    special_token_size = len(special_token)
    least_vocab_size = INIT_VOCAB_SIZE + special_token_size

    if vocab_size < least_vocab_size:
        raise ValueError("vocab_size must be at least ", least_vocab_size)

    for idx in range(INIT_VOCAB_SIZE):
        vocab[idx] = bytes([idx])

    for idx in range(INIT_VOCAB_SIZE, least_vocab_size):
        s_token_id = idx - INIT_VOCAB_SIZE
        s_token = special_token[s_token_id]
        s_token_utf8 = s_token.encode("utf-8")
        vocab[idx] = s_token_utf8

    # find text boundaries
    with open(text_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_precesses, b"<|endoftext|>")

    # split task
    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        task = (text_path, start, end, special_token)
        tasks.append(task)

    # pre-tokenization
    print("[Pre-Tokenization] Start", flush=True)
    stage_start = time.perf_counter()
    if not tasks:
        chunk_counts = []
    elif num_precesses == 1:
        # 单线程就serial执行，避免起线程池
        for task in tasks:
            counts = count_chunk(*task)
            chunk_counts.append(counts)
    else:
        worker_count = min(num_precesses, len(tasks))

        with Pool(processes=worker_count) as pool:
            chunk_counts = pool.starmap(count_chunk, tasks)

    for counts in chunk_counts:
        for fragment, cnt in counts.items():
            if fragment in total_counts:
                total_counts[fragment] += cnt
            else:
                total_counts[fragment] = cnt

    print(
        f"[Pre-Tokenization] End. Time ={time.perf_counter() - stage_start:.2f}s",
        flush=True,
    )

    # convery string to bytes
    for fragment, cnt in total_counts.items():
        frag_utf8 = fragment.encode("utf-8")
        res = []
        for value in frag_utf8:
            byte = bytes([value])
            res.append(byte)
        res_tuple = tuple(res)
        counts_bytes[res_tuple] = cnt

    # BPE merge loop
    bpe_start = time.perf_counter()
    print("[BPE loop] Start", flush=True)

    counts_pair = find_counts_pair(counts_bytes)
    for it in range(vocab_size - least_vocab_size):
        if (it + 1) % 1000 == 0:
            print("[Training] it", it + 1, "/ total ", vocab_size - least_vocab_size)
        if not counts_pair:
            break

        merged_pair = find_merge_pair(counts_pair)
        merges.append(merged_pair)
        counts_bytes = merge(counts_pair, counts_bytes, merged_pair)

    print(
        f"[BPE loop] End. Time={time.perf_counter() - bpe_start:.2f}s",
        flush=True,
    )
    for pair in merges:
        vocab[len(vocab)] = pair[0] + pair[1]

    return vocab, merges


def main():
    text_path = "data/TinyStoriesV2-GPT4-train.txt"
    special_token = ["<|endoftext|>"]
    vocab_size = 10000
    num_precesses = 16
    vocab, merges = train(text_path, vocab_size, special_token, num_precesses)
    print(len(vocab))
    print(len(merges))


if __name__ == "__main__":
    main()
