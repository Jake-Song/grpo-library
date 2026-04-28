r"""
Standalone GRPO training on IFEval using TRL + vLLM (colocate mode).

Single-file trainer. Uses `trl.GRPOTrainer` directly (no Unsloth, no
separate vLLM server) with `vllm_mode="colocate"` so vLLM runs inside
the trainer process on the same GPU. Reward is the strict-pass IFEval
constraint checker (`summarize_constraint_evaluation`), inlined below.

Dataset: `allenai/IF_multi_constraints_upto5` (same as scripts/if_train.py
and scripts/if_train_unsloth.py).

Colab install (paste into a cell BEFORE running this script):

    %%capture
    !pip install --upgrade -qqq uv
    !uv pip install --system -qqq sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer accelerate
    !uv pip install --system -qqq vllm==0.15.1
    !uv pip install --system -qqq transformers==4.56.2
    !uv pip install --system -qqq --no-deps trl==0.22.2

Then run:

    !python if_train_trl.py

Local install (into the trainer env in this repo):

    uv --project envs/trainer add trl vllm accelerate

Local run:

    uv --project envs/trainer run python scripts/if_train_trl.py

Smoke-test the inlined reward checkers (no GPU / TRL / vLLM required):

    python scripts/if_train_trl.py --smoke-test
"""

import ast
import collections
import json
import re
import sys
from typing import Any


# =============================================================================
# IFEval reward checkers — copied verbatim from tasks/IF_EVAL.py
# =============================================================================

CONSTRAINED_RESPONSE_OPTIONS = (
    "My answer is yes.",
    "My answer is no.",
    "My answer is maybe.",
)


def _normalize_kwargs(kw: dict | None) -> dict:
    return kw if isinstance(kw, dict) else {}


def _count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def _word_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)


def _detect_language(text: str) -> str | None:
    try:
        from langdetect import LangDetectException, detect

        try:
            return detect(text)
        except LangDetectException:
            return None
    except ImportError:
        if re.search(r"[Ѐ-ӿ]", text):
            return "ru"
        if re.search(r"[一-鿿]", text):
            return "zh"
        if re.search(r"[A-Za-z]", text):
            return "en"
        return None


def check_no_comma(text: str, kw: dict) -> bool:
    return not re.search(r"\,", text)


def check_number_words(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    relation = kw.get("relation")
    num_words = kw.get("num_words")
    if relation is None or num_words is None:
        return True
    count = _count_words(text)
    if relation == "less than":
        return count < num_words
    if relation == "at least":
        return count >= num_words
    return True


def check_number_sentences(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    relation = kw.get("relation")
    num_sentences = kw.get("num_sentences")
    if relation is None or num_sentences is None:
        return True
    count = len(_split_sentences(text))
    if relation == "less than":
        return count < num_sentences
    if relation == "at least":
        return count >= num_sentences
    return True


def check_number_paragraphs(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    target = kw.get("num_paragraphs")
    if target is None:
        return True
    paragraphs = re.split(r"\s?\*\*\*\s?", text)
    num_paragraphs = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index == 0 or index == len(paragraphs) - 1:
                num_paragraphs -= 1
            else:
                return False
    return num_paragraphs == target


def check_keywords_existence(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    keywords = kw.get("keywords")
    if not keywords:
        return True
    return all(re.search(keyword, text, flags=re.IGNORECASE) for keyword in keywords)


def check_forbidden_words(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    forbidden = kw.get("forbidden_words")
    if not forbidden:
        return True
    return all(not re.search(r"\b" + word + r"\b", text, flags=re.IGNORECASE) for word in forbidden)


def check_keyword_frequency(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    keyword = kw.get("keyword")
    frequency = kw.get("frequency")
    relation = kw.get("relation")
    if keyword is None or frequency is None or relation is None:
        return True
    count = len(re.findall(keyword, text, flags=re.IGNORECASE))
    if relation == "less than":
        return count < frequency
    if relation == "at least":
        return count >= frequency
    return True


def check_english_lowercase(text: str, kw: dict) -> bool:
    try:
        from langdetect import LangDetectException, detect

        try:
            return text.islower() and detect(text) == "en"
        except LangDetectException:
            return True
    except ImportError:
        return text.islower() and _detect_language(text) == "en"


def check_english_capital(text: str, kw: dict) -> bool:
    try:
        from langdetect import LangDetectException, detect

        try:
            return text.isupper() and detect(text) == "en"
        except LangDetectException:
            return True
    except ImportError:
        return text.isupper() and _detect_language(text) == "en"


def check_title(text: str, kw: dict) -> bool:
    return any(title.lstrip("<").rstrip(">").strip() for title in re.findall(r"<<[^\n]+>>", text))


def check_number_highlighted_sections(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    num_highlights = kw.get("num_highlights")
    if num_highlights is None:
        return True
    count = 0
    for highlight in re.findall(r"\*[^\n\*]*\*", text):
        if highlight.strip("*").strip():
            count += 1
    for highlight in re.findall(r"\*\*[^\n\*]*\*\*", text):
        if highlight.removeprefix("**").removesuffix("**").strip():
            count += 1
    return count >= num_highlights


def check_number_bullet_lists(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    num_bullets = kw.get("num_bullets")
    if num_bullets is None:
        return True
    count = len(re.findall(r"^\s*\*[^\*].*$", text, flags=re.MULTILINE))
    count += len(re.findall(r"^\s*-.*$", text, flags=re.MULTILINE))
    return count == num_bullets


def check_json_format(text: str, kw: dict) -> bool:
    candidate = (
        text.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        json.loads(candidate)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def check_postscript(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    marker = kw.get("postscript_marker")
    if not marker:
        return True
    value = text.lower()
    if marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + marker.lower() + r".*$"
    return bool(re.findall(pattern, value, flags=re.MULTILINE))


def check_end_checker(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    end_phrase = kw.get("end_phrase")
    if end_phrase is None:
        return True
    return text.strip().strip('"').lower().endswith(str(end_phrase).strip().lower())


def check_quotation(text: str, kw: dict) -> bool:
    stripped = text.strip()
    return len(stripped) > 1 and stripped[0] == '"' and stripped[-1] == '"'


def check_repeat_prompt(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    prompt_to_repeat = kw.get("prompt_to_repeat")
    if prompt_to_repeat is None:
        return True
    return bool(text.strip().lower().startswith(str(prompt_to_repeat).strip().lower()))


def check_keyword_letter_frequency(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    letter = kw.get("letter")
    let_frequency = kw.get("let_frequency")
    let_relation = kw.get("let_relation")
    if not letter or let_frequency is None or let_relation is None:
        return True
    normalized = str(letter).strip().lower()
    if len(normalized) != 1:
        return False
    count = collections.Counter(text.lower())[normalized]
    if let_relation == "less than":
        return count < let_frequency
    if let_relation == "at least":
        return count >= let_frequency
    return True


def check_response_language(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    language = kw.get("language")
    if not language:
        return True
    try:
        from langdetect import LangDetectException, detect

        try:
            return detect(text) == language
        except LangDetectException:
            return True
    except ImportError:
        detected = _detect_language(text)
        return detected == language if detected else False


def check_nth_paragraph_first_word(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    num_paragraphs = kw.get("num_paragraphs")
    nth_paragraph = kw.get("nth_paragraph")
    first_word = kw.get("first_word")
    if num_paragraphs is None or nth_paragraph is None or first_word is None:
        return True

    paragraphs = re.split(r"\n\n", text)
    total = len(paragraphs)
    for paragraph in paragraphs:
        if not paragraph.strip():
            total -= 1

    if nth_paragraph > total or nth_paragraph <= 0:
        return False
    paragraph = paragraphs[nth_paragraph - 1].strip()
    if not paragraph:
        return False

    extracted = ""
    punctuation = {".", ",", "?", "!", "'", '"'}
    word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
    for letter in word:
        if letter in punctuation:
            break
        extracted += letter.lower()
    return total == num_paragraphs and extracted == str(first_word).lower()


def check_number_placeholders(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    num_placeholders = kw.get("num_placeholders")
    if num_placeholders is None:
        return True
    return len(re.findall(r"\[.*?\]", text)) >= num_placeholders


def check_constrained_response(text: str, kw: dict) -> bool:
    value = text.strip()
    return any(option in value for option in CONSTRAINED_RESPONSE_OPTIONS)


def check_multiple_sections(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    section_spliter = kw.get("section_spliter")
    num_sections = kw.get("num_sections")
    if not section_spliter or num_sections is None:
        return True
    pattern = r"\s?" + str(section_spliter) + r"\s?\d+\s?"
    sections = re.split(pattern, text)
    return max(len(sections) - 1, 0) >= num_sections


def check_two_responses(text: str, kw: dict) -> bool:
    valid_responses = []
    responses = text.split("******")
    for index, response in enumerate(responses):
        if not response.strip():
            if index != 0 and index != len(responses) - 1:
                return False
        else:
            valid_responses.append(response)
    return len(valid_responses) == 2 and valid_responses[0].strip() != valid_responses[1].strip()


def check_capital_word_frequency(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    frequency = kw.get("capital_frequency")
    relation = kw.get("capital_relation")
    if frequency is None or relation is None:
        return True
    words = _word_tokenize(text)
    count = sum(1 for word in words if word.isupper())
    if relation == "less than":
        return count < frequency
    if relation == "at least":
        return count >= frequency
    return True


def check_repeat_phrase(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    phrase = kw.get("phrase")
    small_n = kw.get("small_n")
    if not phrase or small_n is None:
        return True
    tokens = str(phrase).split()
    if not tokens:
        return True
    first_word, last_word = tokens[0], tokens[-1]
    found = re.findall(rf"{re.escape(first_word)} .*? {re.escape(last_word)}", text)
    if len(found) != small_n:
        return False
    ref = tokens
    num_satisfied = 0
    for fragment in found:
        parts = fragment.split()
        if len(parts) != len(ref):
            return False
        diffs = 0
        for a, b in zip(parts, ref):
            if a != b:
                diffs += 1
                if diffs > 1:
                    return False
        num_satisfied += diffs == 1
    return num_satisfied == small_n


def check_copy(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    prompt_to_repeat = kw.get("prompt_to_repeat")
    if prompt_to_repeat is None:
        return True
    return text.strip().lower() == str(prompt_to_repeat).strip().lower()


def check_copying_simple(text: str, kw: dict) -> bool:
    return check_copy(text, kw)


def check_copying_multiple(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    prompt_to_repeat = kw.get("prompt_to_repeat")
    n_copies = kw.get("N")
    if prompt_to_repeat is None or n_copies is None:
        return True
    prompts = text.split("******")
    if len(prompts) != n_copies:
        return False
    target = str(prompt_to_repeat).strip().lower()
    return all(p.strip().lower() == target for p in prompts)


def check_copy_span_idx(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    prompt_to_repeat = kw.get("prompt_to_repeat")
    n_start = kw.get("n_start")
    n_end = kw.get("n_end")
    if prompt_to_repeat is None or n_start is None or n_end is None:
        return True
    span = str(prompt_to_repeat)[n_start:n_end]
    return text.strip().lower() == span.strip().lower()


def check_sentence_hyphens(text: str, kw: dict) -> bool:
    sentences_gold = _split_sentences(re.sub("-", " ", text))
    sentences = text.split("-")
    for sentence, gold in zip(sentences, sentences_gold):
        if sentence.strip() != sentence or sentence != gold:
            return False
    return True


def check_square_brackets(text: str, kw: dict) -> bool:
    words = text.split()
    if not words:
        return False
    return all(word.startswith("[") and word.endswith("]") for word in words)


def check_bigram_wrapping(text: str, kw: dict) -> bool:
    words = text.split()
    for i in range(0, len(words) - 1, 2):
        if i + 1 < len(words) and not (words[i].startswith("<<") and words[i + 1].endswith(">>")):
            return False
    return True


def check_word_once(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    keyword = kw.get("keyword")
    if not keyword:
        return True
    return len(re.findall(re.escape(str(keyword)), text, flags=re.IGNORECASE)) == 1


def check_word_count_different_numbers(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    keyword = kw.get("keyword")
    frequency = kw.get("frequency")
    relation = kw.get("relation")
    if not keyword or frequency is None or relation is None:
        return True
    count = len(re.findall(re.escape(str(keyword)), text, flags=re.IGNORECASE))
    if relation == "less than":
        return count < frequency
    if relation == "at least":
        return count >= frequency
    return True


def check_exclude_word_harder(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    keyword = kw.get("keyword")
    if keyword is None:
        return True
    return " " + str(keyword) + " " not in text


def check_no_adjacent_consecutive(text: str, kw: dict) -> bool:
    words = text.split()
    for i in range(len(words) - 1):
        if not words[i] or not words[i + 1]:
            continue
        a = words[i][0].lower()
        b = words[i + 1][0].lower()
        if not a.isalpha() or not b.isalpha():
            continue
        if ord(b) - ord(a) == 1:
            return False
    return True


def check_palindrome(text: str, kw: dict) -> bool:
    return any(word == word[::-1] for word in text.split() if word)


def check_keyword_specific_position(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    keyword = kw.get("keyword")
    n = kw.get("n")
    m = kw.get("m")
    if keyword is None or n is None or m is None:
        return True
    sentences = _split_sentences(text)
    if len(sentences) < n:
        return False
    words = _word_tokenize(sentences[n - 1])
    if len(words) < m:
        return False
    return words[m - 1] == str(keyword)


def check_start_end(text: str, kw: dict) -> bool:
    words = _word_tokenize(text)
    if len(words) < 2:
        return False
    return words[0].lower() == words[-1].lower()


def check_paragraphs_stars(text: str, kw: dict) -> bool:
    paragraphs = re.split(r"\s?\*\*\*\s?", text)
    num = len(paragraphs)
    for i, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if i == 0 or i == len(paragraphs) - 1:
                num -= 1
            else:
                return False
    return num == 2


def check_paragraphs_double_newline(text: str, kw: dict) -> bool:
    paragraphs = re.split(r"\n\n", text)
    num = len(paragraphs)
    for i, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if i == 0 or i == len(paragraphs) - 1:
                num -= 1
            else:
                return False
    return num == 2


def check_first_word_sent(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    first_word = kw.get("first_word")
    if first_word is None:
        return True
    target = str(first_word).lower()
    sentences = _split_sentences(text)
    if not sentences:
        return False
    for sentence in sentences:
        if not sentence.strip():
            return False
        w = sentence.split()[0].strip()
        if w.lower() != target:
            return False
    return True


def check_first_word_answer(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    first_word = kw.get("first_word")
    if first_word is None:
        return True
    tokens = text.split()
    if not tokens:
        return False
    return tokens[0].strip().lower() == str(first_word).strip().lower()


def check_last_word_sent(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    last_word = kw.get("last_word")
    if last_word is None:
        return True
    target = str(last_word).lower()
    sentences = _split_sentences(text)
    if not sentences:
        return False
    for sentence in sentences:
        if not sentence.strip():
            return False
        w = sentence.split()[-1].strip()
        w = re.sub(r"[^\w\s]", "", w)
        if w.lower() != target:
            return False
    return True


def check_last_word_answer(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    last_word = kw.get("last_word")
    if last_word is None:
        return True
    tokens = text.split()
    if not tokens:
        return False
    w = re.sub(r"[^\w\s]", "", tokens[-1].strip())
    return w.lower() == str(last_word).strip().lower()


def check_punctuation_dot(text: str, kw: dict) -> bool:
    return "." not in text


def check_punctuation_exclamation(text: str, kw: dict) -> bool:
    return "!" not in text


def check_lowercase_counting(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    n = kw.get("N")
    if n is None:
        return True
    lowercase = re.findall(r"\b[a-z]+\b", text)
    return len(lowercase) <= n


def check_count_unique(text: str, kw: dict) -> bool:
    words = _word_tokenize(text)
    if not words:
        return False
    return len(words) == len(set(words))


def check_count_increment_word(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    keyword1 = kw.get("keyword1")
    keyword2 = kw.get("keyword2")
    if keyword1 is None or keyword2 is None:
        return True
    c1 = len(re.findall(re.escape(str(keyword1)), text, flags=re.IGNORECASE))
    c2 = len(re.findall(re.escape(str(keyword2)), text, flags=re.IGNORECASE))
    return c1 == 1 and c2 == 2


def check_counting_composition(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    n_sent = kw.get("n_sent")
    n_words = kw.get("n_words")
    if n_sent is None or n_words is None:
        return True
    paragraphs = re.split(r"\s?\*\*\*\s?", text)
    num = len(paragraphs)
    for i, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if i == 0 or i == len(paragraphs) - 1:
                num -= 1
                continue
            return False
        sentences = _split_sentences(paragraph)
        if len(sentences) != n_sent:
            return False
        for sentence in sentences:
            if len(_word_tokenize(sentence)) != n_words:
                return False
    return num == 3


def check_letter_counting(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    n = kw.get("N")
    relation = kw.get("relation")
    if n is None or relation is None:
        return True
    letters = re.findall(r"[a-zA-Z]", text)
    if relation == "at least":
        return len(letters) >= n
    if relation == "less than":
        return len(letters) < n
    return True


check_letter_counting2 = check_keyword_letter_frequency


CHECKERS = {
    "punctuation:no_comma": check_no_comma,
    "length_constraints:number_words": check_number_words,
    "length_constraints:number_sentences": check_number_sentences,
    "length_constraints:number_paragraphs": check_number_paragraphs,
    "length_constraints:nth_paragraph_first_word": check_nth_paragraph_first_word,
    "keywords:existence": check_keywords_existence,
    "keywords:forbidden_words": check_forbidden_words,
    "keywords:frequency": check_keyword_frequency,
    "keywords:letter_frequency": check_keyword_letter_frequency,
    "language:response_language": check_response_language,
    "change_case:english_lowercase": check_english_lowercase,
    "change_case:english_capital": check_english_capital,
    "change_case:capital_word_frequency": check_capital_word_frequency,
    "detectable_format:title": check_title,
    "detectable_format:number_highlighted_sections": check_number_highlighted_sections,
    "detectable_format:number_bullet_lists": check_number_bullet_lists,
    "detectable_format:constrained_response": check_constrained_response,
    "detectable_format:multiple_sections": check_multiple_sections,
    "detectable_format:json_format": check_json_format,
    "detectable_content:number_placeholders": check_number_placeholders,
    "detectable_content:postscript": check_postscript,
    "startend:end_checker": check_end_checker,
    "startend:quotation": check_quotation,
    "combination:repeat_prompt": check_repeat_prompt,
    "combination:two_responses": check_two_responses,
    "copy:repeat_phrase": check_repeat_phrase,
    "copy:copy": check_copy,
    "copy:copying_simple": check_copying_simple,
    "copy:copying_multiple": check_copying_multiple,
    "new:copy_span_idx": check_copy_span_idx,
    "detectable_format:sentence_hyphens": check_sentence_hyphens,
    "detectable_format:square_brackets": check_square_brackets,
    "detectable_format:bigram_wrapping": check_bigram_wrapping,
    "keywords:word_once": check_word_once,
    "keywords:word_count_different_numbers": check_word_count_different_numbers,
    "keywords:exclude_word_harder": check_exclude_word_harder,
    "keywords:no_adjacent_consecutive": check_no_adjacent_consecutive,
    "keywords:palindrome": check_palindrome,
    "keywords:keyword_specific_position": check_keyword_specific_position,
    "keywords:start_end": check_start_end,
    "paragraphs:paragraphs": check_paragraphs_stars,
    "paragraphs:paragraphs2": check_paragraphs_double_newline,
    "first_word:first_word_answer": check_first_word_answer,
    "first_word:first_word_sent": check_first_word_sent,
    "last_word:last_word_answer": check_last_word_answer,
    "last_word:last_word_sent": check_last_word_sent,
    "punctuation:punctuation_dot": check_punctuation_dot,
    "punctuation:punctuation_exclamation": check_punctuation_exclamation,
    "count:lowercase_counting": check_lowercase_counting,
    "count:count_unique": check_count_unique,
    "count:count_increment_word": check_count_increment_word,
    "count:counting_composition": check_counting_composition,
    "letters:letter_counting": check_letter_counting,
    "letters:letter_counting2": check_letter_counting2,
}


def summarize_constraint_evaluation(
    text: str,
    instruction_ids: list[str],
    kwargs_list: list[dict],
) -> dict[str, Any]:
    stripped = text.strip()
    results: dict[str, bool | None] = {}
    passed = 0
    implemented = 0

    for idx, constraint_id in enumerate(instruction_ids):
        kw = _normalize_kwargs(kwargs_list[idx] if idx < len(kwargs_list) else None)
        checker = CHECKERS.get(constraint_id)
        if checker is None:
            results[constraint_id] = None
            continue
        implemented += 1
        result = checker(stripped, kw)
        results[constraint_id] = result
        if result:
            passed += 1

    all_passed = (
        bool(stripped)
        and bool(instruction_ids)
        and implemented == len(instruction_ids)
        and passed == implemented
    )
    reward = 1.0 if all_passed else 0.0
    return {
        "results": results,
        "reward": reward,
        "implemented": implemented,
        "passed": passed,
        "all_passed": all_passed,
    }


# =============================================================================
# Reward function (TRL GRPOTrainer signature)
# =============================================================================

def ifeval_reward(prompts, completions, instruction_id_list, kwargs_list, **_):
    """Strict-pass IFEval reward: 1.0 iff every constraint passes, else 0.0.

    `completions` is a list of message-lists (one assistant turn each) when
    the dataset's `prompt` column is a chat. `instruction_id_list` and
    `kwargs_list` come straight from dataset columns, batched.
    """
    rewards = []
    for completion, ids, kws in zip(completions, instruction_id_list, kwargs_list):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        rewards.append(summarize_constraint_evaluation(text, ids, kws)["reward"])
    return rewards


# =============================================================================
# Smoke test (no GPU / TRL / vLLM required)
# =============================================================================

def _smoke_test() -> None:
    cases = [
        ("My answer is yes.", ["keywords:existence"], [{"keywords": ["yes"]}], 1.0),
        ("Nope.", ["keywords:existence"], [{"keywords": ["yes"]}], 0.0),
        ("This response ends on brief.", ["last_word:last_word_answer"], [{"last_word": "brief"}], 1.0),
        ("the quick brown fox", ["count:count_unique"], [{}], 1.0),
        ("the the quick brown fox", ["count:count_unique"], [{}], 0.0),
        (
            "First paragraph.\n***\nLast word is brief.",
            ["paragraphs:paragraphs", "last_word:last_word_answer"],
            [{}, {"last_word": "brief"}],
            1.0,
        ),
    ]
    rewards = ifeval_reward(
        prompts=[None] * len(cases),
        completions=[c[0] for c in cases],
        instruction_id_list=[c[1] for c in cases],
        kwargs_list=[c[2] for c in cases],
    )
    for (text, ids, _kws, expected), got in zip(cases, rewards):
        status = "OK" if got == expected else "FAIL"
        print(f"{status} expected={expected} got={got} ids={ids}")
        assert got == expected, (text, ids, expected, got)
    print("Smoke test passed.")


# =============================================================================
# Training pipeline (TRL GRPOTrainer + vLLM colocate)
# =============================================================================

MODEL_NAME = "unsloth/Qwen3.5-0.6B"
DATASET_NAME = "allenai/IF_multi_constraints_upto5"
DATASET_CONFIG = "default"

MAX_SEQ_LENGTH = 4096
MAX_TRAIN_EXAMPLES = 1200
MAX_PROMPT_QUANTILE = 0.9       # filter out the longest 10% of prompts

PER_DEVICE_BATCH_SIZE = 4
NUM_GENERATIONS = 4
GRAD_ACCUM_STEPS = 1
MAX_STEPS = 100
SAVE_STEPS = 100

LEARNING_RATE = 3e-6
TEMPERATURE = 0.7
WEIGHT_DECAY = 0.0
SEED = 3407

OUTPUT_DIR = "outputs/if_train_trl"


def main() -> None:
    import numpy as np
    import torch
    from datasets import Dataset, load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from vllm import SamplingParams

    print(f"Loading tokenizer + model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)

    print(f"Loading dataset {DATASET_NAME}/{DATASET_CONFIG}")
    raw = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")

    rows = []
    for example in raw:
        messages = example.get("messages") or []
        if not messages:
            continue
        prompt_text = (messages[0].get("content") or "").strip()
        if not prompt_text:
            continue
        gt = ast.literal_eval(example["ground_truth"])[0]
        instruction_ids = gt.get("instruction_id") or []
        kwargs_raw = gt.get("kwargs") or []
        kwargs_clean = [kw if isinstance(kw, dict) else {} for kw in kwargs_raw]
        rows.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "instruction_id_list": instruction_ids,
            "kwargs_list": kwargs_clean,
        })
    if not rows:
        raise ValueError(f"No usable rows in {DATASET_NAME}/{DATASET_CONFIG}.")
    if len(rows) > MAX_TRAIN_EXAMPLES:
        rows = rows[:MAX_TRAIN_EXAMPLES]
    dataset = Dataset.from_list(rows)
    print(f"Dataset rows: {len(dataset)}")

    prompt_token_lens = [
        len(tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=True))
        for row in rows
    ]
    max_prompt_length = int(np.quantile(prompt_token_lens, MAX_PROMPT_QUANTILE)) + 1
    keep_idx = [i for i, n in enumerate(prompt_token_lens) if n <= max_prompt_length]
    dataset = dataset.select(keep_idx)
    max_completion_length = MAX_SEQ_LENGTH - max_prompt_length
    print(
        f"max_prompt_length={max_prompt_length} "
        f"max_completion_length={max_completion_length} "
        f"after filter: {len(dataset)} rows"
    )

    vllm_sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=1.0,
        top_k=-1,
        seed=SEED,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=0.0,
        lr_scheduler_type="linear",
        optim="adamw_torch_fused",
        logging_steps=1,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_generations=NUM_GENERATIONS,
        max_completion_length=max_completion_length,
        max_steps=MAX_STEPS,
        save_steps=SAVE_STEPS,
        seed=SEED,
        report_to="none",
        bf16=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.4,
        temperature=TEMPERATURE,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[ifeval_reward],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/final")
    print(f"Saved final model to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        _smoke_test()
        sys.exit(0)
    main()
