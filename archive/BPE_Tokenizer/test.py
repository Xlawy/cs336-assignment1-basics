def main():
    data = b"hi\xff"

    hex_text = data.hex()
    print(hex_text)

    restored = bytes.fromhex(hex_text)
    print(restored)


if __name__ == "__main__":
    main()
