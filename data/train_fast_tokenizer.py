import os
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def gather_corpus_files(src_dir: str, add_chinese: bool = False, chinese_dir: str = "./data/chinese_corpus") -> list:
    """Gather all text files for tokenizer training, optionally including Chinese corpus."""
    files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".txt")]

    if add_chinese:
        ch_path = chinese_dir
        if os.path.isdir(ch_path):
            zh_files = [os.path.join(ch_path, f) for f in os.listdir(ch_path) if f.endswith(".txt")]
            if zh_files:
                print(f"   Adding {len(zh_files)} Chinese corpus files from {ch_path}")
                files.extend(zh_files)
            else:
                print(f"   Warning: No .txt files found in {ch_path} — training on English corpus only")
        else:
            print(f"   Warning: Chinese corpus directory not found at {ch_path}")
            print(f"   Run: python data/pipeline.py to download Chinese corpus, or:")
            print(f"   mkdir -p {ch_path} && place .txt files inside")

    return files


def train_fast_tokenizer(src_dir: str = "./data/super_cleaned_corpus",
                         output_dir: str = "./data",
                         vocab_size: int = 32000,
                         add_chinese: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Initializing Byte-Level BPE Tokenizer ({vocab_size} vocab) training...")

    # 1. Initialize Byte-Level BPE model
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # 2. Setup trainer
    special_tokens = ["<|im_start|>", "<|im_end|>", "<|pad|>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
    )

    # 3. Gather corpus files
    files = gather_corpus_files(src_dir, add_chinese=add_chinese)

    if not files:
        print(f"Error: No text files found in {src_dir}. Please run clean_unk.py first.")
        return

    print(f"   Training BPE on {len(files)} files...")
    tokenizer.train(files, trainer)

    # 4. Save tokenizer
    output_path = os.path.join(output_dir, "fast_tokenizer.json")
    tokenizer.save(output_path)
    print(f"Fast BPE Tokenizer trained successfully! Saved to: {output_path}")

    # 5. English round-trip verification
    test_en = "Deep learning breakthrough: Albert Einstein invented BPE tokenization in 1957!"
    encoded_en = tokenizer.encode(test_en)
    decoded_en = tokenizer.decode(encoded_en.ids)
    print("=" * 60)
    print("English Tokenizer Verification:")
    print(f"   Input:   '{test_en}'")
    print(f"   Tokens:  {len(encoded_en.ids)} tokens")
    print(f"   IDs:     {encoded_en.ids[:20]}...")
    print(f"   Decoded: '{decoded_en}'")
    print(f"   Match:   {test_en == decoded_en}")

    # 6. Chinese encoding efficiency test
    test_zh = "深度学习彻底改变了自然语言处理领域的研究范式"
    encoded_zh = tokenizer.encode(test_zh)
    decoded_zh = tokenizer.decode(encoded_zh.ids)
    zh_chars = len(test_zh)
    zh_tokens = len(encoded_zh.ids)
    efficiency = zh_chars / max(zh_tokens, 1)
    print("-" * 60)
    print("Chinese Encoding Efficiency Test:")
    print(f"   Input:       '{test_zh}'")
    print(f"   Characters:  {zh_chars}")
    print(f"   Tokens:      {zh_tokens}")
    print(f"   Efficiency:  {efficiency:.2f} chars/token (higher = better)")
    print(f"   Token IDs:   {encoded_zh.ids}")
    print(f"   Decoded:     '{decoded_zh}'")
    print(f"   Round-trip:  {test_zh == decoded_zh}")
    if efficiency < 1.0:
        print(f"   Warning: Chinese encoding efficiency is low ({efficiency:.2f} chars/token).")
        print(f"   Consider increasing vocab_size or adding more Chinese training data.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Byte-Level BPE tokenizer for nano-llm")
    parser.add_argument("--corpus_dir", type=str, default="./data/super_cleaned_corpus",
                        help="Directory containing cleaned English text corpus")
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="Target vocabulary size")
    parser.add_argument("--output", type=str, default="./data/fast_tokenizer.json",
                        help="Output path for trained tokenizer")
    parser.add_argument("--add_chinese", action="store_true",
                        help="Mix Chinese text corpus into training data")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    train_fast_tokenizer(
        src_dir=args.corpus_dir,
        output_dir=output_dir,
        vocab_size=args.vocab_size,
        add_chinese=args.add_chinese,
    )
