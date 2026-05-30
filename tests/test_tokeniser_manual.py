"""Manual test script for SubwordTokeniser using real dataset.

This script:
1. Loads the Chanchal et al. dataset (downloads if missing)
2. Trains a BPE tokeniser on the text data
3. Tests encoding, decoding, and persistence
4. Demonstrates batch encoding
"""

from src.dataset import DatasetLoader
from src.tokeniser import SubwordTokeniser

print("=" * 70)
print("SubwordTokeniser Manual Test with Real Dataset")
print("=" * 70)

# Load dataset (will download if missing)
print("\n1. Loading dataset...")
loader = DatasetLoader()
texts, labels = loader.load(
    "data/AuthorIdentification/Dataset/Dataset_with_varying_number_of_tweets/50_tweets_per_user.csv",
    fetch_if_missing=True
)
print(f"   Loaded {len(texts)} samples from {loader.num_authors} authors")

# Train tokeniser on first 1000 texts (faster for demo)
print("\n2. Training BPE tokeniser...")
train_texts = texts[:1000]
tok = SubwordTokeniser()
tok.train(train_texts, vocab_size=5000, algorithm="bpe")
print(f"   Trained tokeniser with vocab size: {tok.vocab_size()}")

# Test encoding
print("\n3. Testing encode()...")
sample_text = texts[0]
print(f"   Original text: {sample_text[:100]}...")
max_length = 64
ids = tok.encode(sample_text, max_length=max_length)
print(f"   Encoded to {len(ids)} token IDs: {ids[:20]}...")
print(f"   PAD tokens: {ids.count(0)}")

# Test decoding
print("\n4. Testing decode()...")
decoded = tok.decode(ids)
print(f"   Decoded text: {decoded[:100]}...")

# Test batch encoding
print("\n5. Testing batch_encode()...")
batch_texts = texts[:5]
batch_ids = tok.batch_encode(batch_texts, max_length=max_length)
print(f"   Batch shape: {batch_ids.shape}")
print(f"   Batch dtype: {batch_ids.dtype}")

# Test save/load
print("\n6. Testing save() and load()...")
save_path = "artifacts/test_tokeniser_manual.json"
tok.save(save_path)
print(f"   Saved to: {save_path}")

tok2 = SubwordTokeniser()
tok2.load(save_path)
print(f"   Loaded tokeniser with vocab size: {tok2.vocab_size()}")

# Verify loaded tokeniser produces same output
ids2 = tok2.encode(sample_text, max_length=max_length)
assert ids == ids2, "Loaded tokeniser produces different IDs!"
print("   ✓ Loaded tokeniser produces identical output")

# Test WordPiece algorithm
print("\n7. Testing WordPiece algorithm...")
tok_wp = SubwordTokeniser()
tok_wp.train(train_texts[:500], vocab_size=3000, algorithm="wordpiece")
print(f"   WordPiece vocab size: {tok_wp.vocab_size()}")
ids_wp = tok_wp.encode(sample_text, max_length=32)
print(f"   WordPiece encoded: {ids_wp[:15]}...")

print("\n" + "=" * 70)
print("All manual tests passed! ✓")
print("=" * 70)
