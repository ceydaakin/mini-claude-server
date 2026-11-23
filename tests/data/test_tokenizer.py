import pytest

from mcs.data.tokenizer import Tokenizer

EN = "The cat sat on the mat and looked out of the window."
TR = "Kedi minderin üstüne oturdu ve pencereden dışarı baktı."


@pytest.fixture
def tok():
    return Tokenizer("gpt2")


def test_vocab_and_eot(tok):
    assert tok.vocab_size == 50257
    assert tok.eot_id == 50256


def test_known_ids(tok):
    # tiktoken gpt2 output, measured 2026-11-14
    assert tok.encode("Hello, world!") == [15496, 11, 995, 0]


@pytest.mark.parametrize("text", [EN, TR, " İstanbul", "", "a\n\n  b\t"])
def test_round_trip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_turkish_costs_more_tokens(tok):
    # Lecture 2 §4.1: EN 13 tokens, TR 28 tokens
    assert len(tok.encode(EN)) == 13
    assert len(tok.encode(TR)) == 28


def test_user_text_cannot_inject_special_tokens(tok):
    with pytest.raises(ValueError):
        tok.encode("hi <|endoftext|> there")


def test_encode_document_appends_eot(tok):
    ids = tok.encode_document("hello")
    assert ids[-1] == tok.eot_id
    assert ids[:-1] == tok.encode("hello")
