#textcleaner.py
from gensim.parsing.preprocessing import preprocess_documents
from jr_abst.summarization.syntactic_unit import SyntacticUnit
from jr_abst.utils import tokenize, has_pattern
from six.moves import range
import re
import logging

logger = logging.getLogger(__name__)
HAS_PATTERN = has_pattern()
if HAS_PATTERN:
    from pattern.en import tag

SEPARATOR = r'@'
RE_SENTENCE = re.compile(r'(\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)', re.UNICODE)
AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)\s(\w)', re.UNICODE)
AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)\s(\w)', re.UNICODE)
AB_ACRONYM_LETTERS = re.compile(r'([a-zA-Z])\.([a-zA-Z])\.', re.UNICODE)
UNDO_AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)' + SEPARATOR + r'(\w)', re.UNICODE)
UNDO_AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)' + SEPARATOR + r'(\w)', re.UNICODE)

def split_sentences(text):
    processed = replace_abbreviations(text)
    return [undo_replacement(sentence) for sentence in get_sentences(processed)]


def replace_abbreviations(text):
    return replace_with_separator(text, SEPARATOR, [AB_SENIOR, AB_ACRONYM])


def undo_replacement(sentence):
    return replace_with_separator(sentence, r" ", [UNDO_AB_SENIOR, UNDO_AB_ACRONYM])


def replace_with_separator(text, separator, regexs):
    replacement = r"\1" + separator + r"\2"
    result = text
    for regex in regexs:
        result = regex.sub(replacement, result)
    return result


def get_sentences(text):
    for match in RE_SENTENCE.finditer(text):
        yield match.group()


def merge_syntactic_units(original_units, filtered_units, tags=None):
    units = []
    for i in range(len(original_units)):
        if filtered_units[i] == '':
            continue
        text = original_units[i]
        token = filtered_units[i]
        tag = tags[i][1] if tags else None
        sentence = SyntacticUnit(text, token, tag, i)
        units.append(sentence)
    return units


def join_words(words, separator=" "):
    return separator.join(words)


def clean_text_by_sentences(text):
    original_sentences = split_sentences(text)
    filtered_sentences = [join_words(sentence) for sentence in preprocess_documents(original_sentences)]
    return merge_syntactic_units(original_sentences, filtered_sentences)


def clean_text_by_word(text, deacc=True):
    text_without_acronyms = replace_with_separator(text, "", [AB_ACRONYM_LETTERS])
    original_words = list(tokenize(text_without_acronyms, to_lower=True, deacc=deacc))
    filtered_words = [join_words(word_list, "") for word_list in preprocess_documents(original_words)]
    if HAS_PATTERN:
        tags = tag(join_words(original_words))  # tag needs the context of the words in the text
    else:
        tags = None
    units = merge_syntactic_units(original_words, filtered_words, tags)
    return {unit.text: unit for unit in units}


def tokenize_by_word(text):
    text_without_acronyms = replace_with_separator(text, "", [AB_ACRONYM_LETTERS])
    return tokenize(text_without_acronyms, to_lower=True, deacc=True)