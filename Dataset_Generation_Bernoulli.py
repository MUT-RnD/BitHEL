import os
import math
import random
import csv
from bitarray import bitarray
from collections import Counter
from scipy.optimize import minimize_scalar

# ==== Utility Functions ====


def bitarray_to_bin_str(ba: bitarray) -> str:
    return ba.to01()


def bin_str_to_bitarray(s: str) -> bitarray:
    return bitarray(s)


def shannon_entropy(bits: bitarray) -> float:
    if not bits:
        return 0.0
    counts = Counter(bits)
    total = len(bits)
    entropy = 0.0
    for bit in (0, 1):
        p = counts.get(bit, 0) / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def p_from_entropy(target_entropy: float) -> float:
    if target_entropy <= 0.0:
        return 0.0
    if target_entropy >= 1.0:
        return 0.5
    result = minimize_scalar(
        lambda p: (binary_entropy(p) - target_entropy) ** 2,
        bounds=(0.001, 0.999),
        method='bounded'
    )
    return result.x


def generate_bernoulli_bitarray(length: int, target_entropy: float) -> bitarray:
    p = p_from_entropy(target_entropy)
    bits = random.choices([0, 1], weights=[1 - p, p], k=length)
    return bitarray(bits)


# ==== Parameters ====

length_COM = [2, 4, 8, 16, 20, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
entropy_levels = [round(e, 2) for e in [i / 100 for i in range(101)]]
samples_per_level = 1000
output_dir = "entropy_tables_csv"
os.makedirs(output_dir, exist_ok=True)

# ==== Generation and Saving ====

print("Generating and saving binary data samples as CSV...")
for length in length_COM:
    for entropy in entropy_levels:
        result_table = []
        print("Length:", length, "Entropy:", entropy)

        if entropy < 0.1 or length < 256:
            for _ in range(samples_per_level):
                result_table.append(generate_bernoulli_bitarray(length, entropy).to01())
        else:
            generated_set = set()
            while len(result_table) < samples_per_level:
                ba = generate_bernoulli_bitarray(length, entropy)
                ba_str = ba.to01()
                if ba_str not in generated_set:
                    generated_set.add(ba_str)
                    result_table.append(ba_str)

        # ✅ Compute and print average entropy
        entropies = [shannon_entropy(bitarray(bs)) for bs in result_table]
        avg_entropy = sum(entropies) / len(entropies)
        print("Length:", length, "Target entropy:", entropy, f"📊 Real avg entropy: {avg_entropy:.4f}")

        filename = f"entropy_table_len{length}_H{entropy:.2f}.csv"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for line in result_table:
                writer.writerow([line])
print("✅ CSV generation complete.\n")


# ==== Loading and Testing ====

print("Loading and testing generated CSV data...")
failed_tests = 0
total_files = 0

for filename in os.listdir(output_dir):
    if filename.endswith(".csv"):
        total_files += 1
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            bitstrings = [row[0] for row in reader if row]

        if len(bitstrings) != samples_per_level:
            print(f"❌ {filename}: Expected {samples_per_level} entries, got {len(bitstrings)}")
            failed_tests += 1
            continue

        length_val = len(bitstrings[0])
        entropy_val = float(filename.split("_H")[1].replace(".csv", ""))
        entropies = [shannon_entropy(bitarray(bs)) for bs in bitstrings]
        avg_entropy = sum(entropies) / len(entropies)
        if entropy_val >= 0.1 and length_val >= 256:
            if len(set(bitstrings)) != samples_per_level:
                print(f"❌ {filename} | 📊 Avg entropy: {avg_entropy:.4f} Not all entries are unique")
                failed_tests += 1
                continue

        # Compute average entropy
        print(f"✅ {filename} | 📊 Avg entropy: {avg_entropy:.4f}")

print(f"\n🧪 Test Summary: {total_files - failed_tests} passed / {total_files} total.")
