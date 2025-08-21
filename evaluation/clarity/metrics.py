import re
import math
from .domain_glossary import is_domain_term

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENT_END_RE = re.compile(r"[.!?]+")


def count_sentences(text: str) -> int:
    if not text:
        return 0
    # heuristic: number of sentence end punctuation groups
    sentences = SENT_END_RE.findall(text)
    if sentences:
        return max(1, len(sentences))
    # fallback: split by newline
    parts = [p for p in re.split(r"[\n\r]+", text) if p.strip()]
    return max(1, len(parts))


def words(text: str):
    return WORD_RE.findall(text or "")


def count_words(text: str) -> int:
    return len(words(text))


def count_syllables_in_word(word: str) -> int:
    word = word.lower()
    if len(word) <= 3:
        return 1
    # simple heuristic for syllable counting
    word = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", word)
    word = re.sub(r"^y", "", word)
    syllables = re.findall(r"[aeiouy]{1,2}", word)
    count = len(syllables)
    return max(1, count)


def count_syllables(text: str) -> int:
    return sum(count_syllables_in_word(w) for w in words(text))


def count_complex_words(text: str) -> int:
    """
    Count complex words (3+ syllables) excluding domain-specific terms.
    Domain terms are excluded from complexity penalty to avoid penalizing
    appropriate technical vocabulary.
    """
    complex_count = 0
    for word in words(text):
        # Skip domain-specific terms - they're not "complex" in context
        if is_domain_term(word):
            continue
        # Count as complex if 3+ syllables
        if count_syllables_in_word(word) >= 3:
            complex_count += 1
    return complex_count


def count_domain_excluded_syllables(text: str) -> int:
    """
    Count syllables excluding domain-specific terms from the count.
    This provides a more accurate syllable count for readability metrics
    when domain expertise is expected.
    """
    total_syllables = 0
    for word in words(text):
        if not is_domain_term(word):
            total_syllables += count_syllables_in_word(word)
        else:
            # For domain terms, use a normalized syllable count (2)
            # to avoid penalizing necessary technical vocabulary
            total_syllables += 2
    return total_syllables


def flesch_reading_ease(text: str, use_domain_exclusion: bool = True) -> float:
    """
    Flesch Reading Ease score with optional domain term handling.
    
    Args:
        text: Text to analyze
        use_domain_exclusion: If True, uses domain-aware syllable counting
        
    Returns:
        Flesch Reading Ease score (0-100, higher = easier)
    """
    w = count_words(text)
    s = count_sentences(text)
    
    if use_domain_exclusion:
        syll = count_domain_excluded_syllables(text)
    else:
        syll = count_syllables(text)
    
    if w == 0 or s == 0:
        return 0.0
    
    score = 206.835 - 1.015 * (w / s) - 84.6 * (syll / w)
    return round(score, 2)


def flesch_kincaid_grade(text: str, use_domain_exclusion: bool = True) -> float:
    """
    Flesch-Kincaid Grade Level with optional domain term handling.
    
    Args:
        text: Text to analyze
        use_domain_exclusion: If True, uses domain-aware syllable counting
        
    Returns:
        Grade level (lower = easier to read)
    """
    w = count_words(text)
    s = count_sentences(text)
    
    if use_domain_exclusion:
        syll = count_domain_excluded_syllables(text)
    else:
        syll = count_syllables(text)
    
    if w == 0 or s == 0:
        return 0.0
    
    grade = 0.39 * (w / s) + 11.8 * (syll / w) - 15.59
    return round(grade, 2)


def smog_index(text: str, use_domain_exclusion: bool = True) -> float:
    """
    SMOG Index with optional domain term handling.
    
    Args:
        text: Text to analyze
        use_domain_exclusion: If True, excludes domain terms from polysyllable count
        
    Returns:
        SMOG index (grade level)
    """
    s = count_sentences(text)
    
    if use_domain_exclusion:
        # Count polysyllables excluding domain terms
        polysyllables = sum(1 for w in words(text) 
                          if not is_domain_term(w) and count_syllables_in_word(w) >= 3)
    else:
        polysyllables = count_complex_words(text)
    
    if s == 0:
        return 0.0
    # scale to 30 sentences
    try:
        score = 1.0430 * math.sqrt(polysyllables * (30.0 / s)) + 3.1291
    except Exception:
        score = 0.0
    return round(score, 2)


def gunning_fog_index(text: str, use_domain_exclusion: bool = True) -> float:
    """
    Gunning Fog Index with optional domain term handling.
    
    Args:
        text: Text to analyze
        use_domain_exclusion: If True, excludes domain terms from complex word count
        
    Returns:
        Gunning Fog index (grade level)
    """
    w = count_words(text)
    s = count_sentences(text)
    
    if use_domain_exclusion:
        complex_words = sum(1 for word in words(text) 
                          if not is_domain_term(word) and count_syllables_in_word(word) >= 3)
    else:
        complex_words = count_complex_words(text)
    
    if w == 0 or s == 0:
        return 0.0
    score = 0.4 * ((w / s) + 100 * (complex_words / w))
    return round(score, 2)


def get_domain_analysis(text: str) -> dict:
    """
    Analyze domain-specific characteristics of text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with domain analysis metrics
    """
    from .domain_glossary import get_domain_term_count, get_domain_coverage
    
    word_list = words(text)
    total_words = len(word_list)
    domain_terms = get_domain_term_count(text)
    domain_coverage = get_domain_coverage(text)
    
    # Count complex words with and without domain exclusion
    complex_all = sum(1 for w in word_list if count_syllables_in_word(w) >= 3)
    complex_non_domain = sum(1 for w in word_list 
                           if not is_domain_term(w) and count_syllables_in_word(w) >= 3)
    
    return {
        'total_words': total_words,
        'domain_terms': domain_terms,
        'domain_coverage_percent': domain_coverage,
        'complex_words_all': complex_all,
        'complex_words_non_domain': complex_non_domain,
        'domain_terms_excluded': complex_all - complex_non_domain
    }
