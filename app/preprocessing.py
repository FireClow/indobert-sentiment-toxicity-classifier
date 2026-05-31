import re
import string
import unicodedata
from typing import Dict


DEFAULT_SLANG_MAP: Dict[str, str] = {
    "gk": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "yg": "yang",
    "dgn": "dengan",
    "krn": "karena",
    "dr": "dari",
    "aja": "saja",
    "bgt": "banget",
    "tp": "tapi",
}

DEFAULT_STOPWORDS = {
    "yang",
    "dan",
    "di",
    "ke",
    "dari",
    "untuk",
    "dengan",
    "atau",
}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
MULTISPACE_PATTERN = re.compile(r"\s+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _remove_punctuation(text: str) -> str:
    punctuation_to_remove = string.punctuation.replace("'", "")
    translator = str.maketrans("", "", punctuation_to_remove)
    return text.translate(translator)


def _normalize_slang(text: str, slang_map: Dict[str, str]) -> str:
    tokens = text.split()
    normalized_tokens = [slang_map.get(token, token) for token in tokens]
    return " ".join(normalized_tokens)


def _remove_stopwords(text: str, stopwords: set[str]) -> str:
    tokens = [token for token in text.split() if token not in stopwords]
    return " ".join(tokens)


def preprocess_text(
    text: str,
    slang_map: Dict[str, str] | None = None,
    remove_stopwords: bool = False,
    stopwords: set[str] | None = None,
) -> str:
    """
    Regex-based preprocessing for Indonesian social media comments.

    Steps:
    1. lowercase
    2. remove URLs, mentions, hashtags, emojis
    3. remove punctuation (except apostrophe)
    4. normalize slang words
    5. optional stopword removal
    """
    if not isinstance(text, str):
        text = ""

    text = _normalize_unicode(text).lower().strip()
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(" ", text)
    text = EMOJI_PATTERN.sub(" ", text)
    text = _remove_punctuation(text)
    text = MULTISPACE_PATTERN.sub(" ", text).strip()

    slang_dict = DEFAULT_SLANG_MAP if slang_map is None else slang_map
    text = _normalize_slang(text, slang_dict)

    if remove_stopwords:
        active_stopwords = DEFAULT_STOPWORDS if stopwords is None else stopwords
        text = _remove_stopwords(text, active_stopwords)

    return MULTISPACE_PATTERN.sub(" ", text).strip()

