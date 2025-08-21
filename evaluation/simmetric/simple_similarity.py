import re
import math
from collections import Counter

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _terms(text: str):
    return [w.lower() for w in WORD_RE.findall(text or "")]


def _tf_vector(text: str):
    terms = _terms(text)
    return Counter(terms)


def cosine_similarity(a: str, b: str) -> float:
    """Compute cosine similarity between two texts using term frequency vectors.
    Returns a float between 0 and 1.
    """
    if not a or not b:
        return 0.0
    va = _tf_vector(a)
    vb = _tf_vector(b)
    # dot product
    common = set(va.keys()) & set(vb.keys())
    dot = sum(va[t] * vb[t] for t in common)
    mag_a = math.sqrt(sum(v * v for v in va.values()))
    mag_b = math.sqrt(sum(v * v for v in vb.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return round(dot / (mag_a * mag_b), 4)
