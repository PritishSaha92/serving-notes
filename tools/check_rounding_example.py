"""Check the article's finite E4M3 example without a GPU or torch."""
from pathlib import Path
import json
import struct
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[1]


def bf16_value(bits):
    return struct.unpack('>f', struct.pack('>I', bits << 16))[0]


def positive_e4m3(bits):
    exponent, fraction = bits >> 3, bits & 7
    return (fraction / 8) * 2**-6 if exponent == 0 else (1 + fraction / 8) * 2**(exponent - 7)


def nearest_e4m3(value):
    # Finite positive codes only. Ties choose an even low significand bit.
    return min(range(127), key=lambda bits: (abs(value - positive_e4m3(bits)), bits & 1))


def main():
    record = json.loads((ROOT / 'evidence/rounding-observation.json').read_text(encoding='utf-8'))
    attempts = record['observation']['attempts']
    assert len(attempts) == 3
    for attempt in attempts:
        assert attempt['query_mismatch_bytes'] == 1
        sample = attempt['samples'][0]
        assert (sample['token'], sample['column']) == (404, 627)
        assert sample['stock_bf16'] == bf16_value(0x3F57) == 0.83984375
        assert sample['reference'] == positive_e4m3(0x35) == 0.8125
        assert sample['candidate'] == positive_e4m3(0x36) == 0.875
    neighbor = bf16_value(0x3F58)
    assert neighbor == 0.84375 == (positive_e4m3(0x35) + positive_e4m3(0x36)) / 2
    assert nearest_e4m3(bf16_value(0x3F57)) == 0x35
    assert nearest_e4m3(neighbor) == 0x36
    assert (bf16_value(0x3F57) - 0.8125) / (0.875 - 0.8125) == 0.4375
    class ByteReader(HTMLParser):
        def __init__(self):
            super().__init__(); self.current = None; self.rows = {}
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'div' and attrs.get('class') == 'byte-row':
                self.current = attrs['data-byte']; self.rows[self.current] = ''
        def handle_data(self, data):
            if self.current is not None:
                self.rows[self.current] += data.strip()
        def handle_endtag(self, tag):
            if tag == 'div': self.current = None
    reader = ByteReader()
    reader.feed((ROOT / 'article.md').read_text(encoding='utf-8'))
    assert reader.rows == {'35': '00110101', '36': '00110110'}
    assert (0x35 ^ 0x36).bit_count() == 2
    print('PASS: saved three-repeat observation, BF16 adjacency, E4M3 ties-to-even, number-line position and displayed bit fields.')


if __name__ == '__main__':
    main()
