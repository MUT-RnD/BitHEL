from bitarray import bitarray
from bitarray.util import int2ba, ba2int
import math
import random
from itertools import permutations
from scipy.optimize import minimize_scalar
from datetime import datetime
from collections import Counter


# === Compression Maps ===
def generate_maps():
    def _build_mapping(a, b, c, d, mode):
        pr = ["010", "011"]
        zr = ["1001", "1000", "0000", "0001", "0010", "0011", "1100", "1101", "1110", "1111"]
        nr = ["10100", "10101", "10110", "10111"]
        p_patterns = [a + a, a + b]
        z_patterns = [a + c, a + d, b + a, b + b, b + c, b + d, c + a, c + b, d + a, d + b]
        n_patterns = [c + c, c + d, d + c, d + d]
        mapping_inside = {}
        key_lengths_inside = {}
        if mode == "compress":
            for key, code in zip(p_patterns, pr):
                mapping_inside[key] = code
            for key, code in zip(z_patterns, zr):
                mapping_inside[key] = code
            for key, code in zip(n_patterns, nr):
                mapping_inside[key] = code
        else:
            for code, key in zip(pr, p_patterns):
                mapping_inside[code] = key
            for code, key in zip(zr, z_patterns):
                mapping_inside[code] = key
            for code, key in zip(nr, n_patterns):
                mapping_inside[code] = key
            key_lengths_inside = {pattern: len(pattern) for pattern in mapping_inside}
        return mapping_inside, key_lengths_inside

    abcd_table = list(permutations(["00", "01", "10", "11"]))[::2]
    mapping_out, inverse_mapping_out, key_lengths_out = [], [], []
    for compress_mod in range(12):
        a, b, c, d = abcd_table[compress_mod]
        m, kl = _build_mapping(a, b, c, d, "compress")
        im, kl = _build_mapping(a, b, c, d, "decompress")
        mapping_out.append(m)
        inverse_mapping_out.append(im)
        key_lengths_out.append(kl)
    return mapping_out, inverse_mapping_out, key_lengths_out


mapping, inverse_mapping, key_lengths = generate_maps()


# === Compression ===
def compress_mode_str(s: str, compress_mod=0) -> str:
    map4 = mapping[compress_mod]
    out = []
    append = out.append
    i = 0

    while i < len(s):
        chunk = s[i:i + 4]
        if len(chunk) < 4:
            append(s[i:])
            break
        code = map4.get(chunk)
        if code is not None:
            append(code)
            i += 4
        else:
            i += 1

    return ''.join(out)


def compress_bits_ba(bits_in: bitarray, value_len: int = 8192) -> bitarray:
    bit_test = bits_in[:value_len].to01()
    best_mode = 12
    best_result = None
    best_len = value_len - 2

    for mode in range(12):
        encoded = compress_mode_str(bit_test, mode)
        elen = len(encoded)
        if elen < best_len:
            best_mode = mode
            best_result = encoded
            best_len = elen

    if best_mode == 12:
        return bitarray('11') + bits_in[:value_len]
    else:
        return int2ba(best_mode, length=4) + bitarray(best_result)


# === Decompression ===
def decompress_mode_str(s: str, decompress_mod=0) -> str:
    inv_map = inverse_mapping[decompress_mod]
    key_lens = sorted(set(key_lengths[decompress_mod].values()))
    out = []
    append = out.append
    i = 0

    while i < len(s):
        for length in key_lens:
            chunk = s[i:i + length]
            if len(chunk) < length:
                continue
            pat = inv_map.get(chunk)
            if pat is not None:
                append(pat)
                i += length
                break
        else:
            append(s[i])
            i += 1

    return ''.join(out)


def decompress_bits_ba(bits_in: bitarray, data_len: int = 8192) -> tuple[bitarray, bitarray]:
    if bits_in[:2] == bitarray('11'):
        return bits_in[2:2 + data_len], bits_in[2 + data_len:]

    mode = ba2int(bits_in[:4])
    encoded = bits_in[4:].to01()
    max_len = len(encoded)
    slice_len = max_len

    for gain in range(max_len):
        try_len = data_len - gain
        if try_len <= 0:
            break
        decoded = decompress_mode_str(encoded[:try_len], mode)
        if len(decoded) == data_len:
            slice_len = try_len
            break

    # final re-decode
    decoded = decompress_mode_str(encoded[:slice_len], mode)
    return bitarray(decoded), bits_in[4 + slice_len:]


# === Supporting Functions ===


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


# === Benchmark Runner ===
def benchmark():
    timer_start = datetime.now()
    print("Start:", timer_start.strftime("%Y/%m/%d %H:%M:%S"))

    #### Benchmark parameters to change
    length_com = 256
    entropy_lvl = 0.98
    repeats = 200
    ####

    gain_sum = ent_in_sum = ent_out_sum = 0
    success = zero_gain = total_success_processes = 0
    print("Analyzed bit string length", length_com, "entropy level of", entropy_lvl, "number of repetitions", repeats)

    for _ in range(repeats):
        while True:
            ct = generate_bernoulli_bitarray(length_com, entropy_lvl)
            se = shannon_entropy(ct)
            # For limited patterns under small length and low entropy
            if entropy_lvl < 0.1 or length_com < 256:
                if abs(se - entropy_lvl) <= 0.03:
                    break
            else:
                break
        c1 = d1 = d2 = bitarray('')

        c1 = compress_bits_ba(ct, length_com)
        test = c1.copy()
        # Ensuring the proper course of the decompression process
        if len(c1) < length_com:
            test += bitarray(''.join(random.choice('01') for _ in range(length_com - len(c1))))
            print("Number of bits padded to the full analyzed length:", length_com - len(c1))

        d1, d2 = decompress_bits_ba(test, length_com)
        same_string = ct == d1
        if same_string:
            total_success_processes += 1
        else:
            print("\n FAILURE!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            print("ct", ct)
            print("d1", d1)
            print("d2", d2)
            print("c1", c1)
            print("t", test)

            if len(test) > len(ct):
                test = False
            else:
                test = True

            print(len(ct), '->', len(c1), '->', len(d1), '+', len(d2), "Restored:", same_string, "Compressed:", test)


        ent_in_sum += se
        ent_out_sum += shannon_entropy(c1)
        gain = len(ct) - len(c1)
        gain_sum += gain
        if gain > 0: success += 1
        elif gain == 0: zero_gain += 1

    total_time = datetime.now() - timer_start
    print("A bit sequence length of", length_com, "was analyzed for an entropy level of", entropy_lvl, "and", repeats, "repetitions.")
    print("Elapsed time:", total_time, "--> Avg time:", total_time / repeats)
    print("Avg input entropy:", ent_in_sum / repeats)
    print("Avg compressed entropy:", ent_out_sum / repeats)
    print("Compression ratio:", ((repeats * length_com) / ((repeats * length_com) - gain_sum)))
    print("Compression saving (%):", (1 - ((repeats * length_com) - gain_sum) / (repeats * length_com)) * 100)
    print("Avg compression gain (bits):", gain_sum / repeats)
    print("Total gain:", gain_sum, "out of", (repeats * length_com), "bits")
    print("Total successfully processed:", total_success_processes, "out of", repeats, "patterns")
    print("Successes:", success, "Zero gain:", zero_gain, "Failures:", repeats - success - zero_gain)


benchmark()
