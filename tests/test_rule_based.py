import pytest
from tocount import estimate_text_tokens, TextEstimator
from tocount.params import INVALID_TEXT_MESSAGE, INVALID_TEXT_ESTIMATOR_MESSAGE


def test_universal_text_with_simple_prompt():
    message = "You are the text completion model"  # https://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=2
    assert isinstance(estimate_text_tokens(message, TextEstimator.RULE_BASED.UNIVERSAL), int)
    assert estimate_text_tokens(message, TextEstimator.RULE_BASED.UNIVERSAL) == pytest.approx(6, abs=5)


def test_universal_text_with_contractions():
    # https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=0
    message = "I’m refining a foolproof method for reality shifting"
    assert isinstance(estimate_text_tokens(message, TextEstimator.RULE_BASED.UNIVERSAL), int)
    assert estimate_text_tokens(message, TextEstimator.RULE_BASED.UNIVERSAL) == pytest.approx(12, abs=10)


def test_universal_text_with_prefixes_and_suffixes():  # https://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=10
    message = "reflecting the hardships of the preparation process"
    assert isinstance(estimate_text_tokens(message, TextEstimator.RULE_BASED.UNIVERSAL), int)
    assert estimate_text_tokens(message, TextEstimator.RULE_BASED.UNIVERSAL) == pytest.approx(8, abs=24)


def test_universal_code_with_keywords():
    # http://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=19
    message1 = "def __init__(self, schema):"
    assert isinstance(estimate_text_tokens(message1, TextEstimator.RULE_BASED.UNIVERSAL), int)
    assert estimate_text_tokens(message1, TextEstimator.RULE_BASED.UNIVERSAL) == pytest.approx(9, abs=5)

    message2 = "class QueryPlanner:"  # http://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=19
    assert isinstance(estimate_text_tokens(message2, TextEstimator.RULE_BASED.UNIVERSAL), int)
    assert estimate_text_tokens(message2, TextEstimator.RULE_BASED.UNIVERSAL) == pytest.approx(5, abs=2)

    message3 = """
    for op in operations:
        if op.type == "SELECT":
    """  # http://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=19
    assert isinstance(estimate_text_tokens(message3, TextEstimator.RULE_BASED.UNIVERSAL), int)
    assert estimate_text_tokens(message3, TextEstimator.RULE_BASED.UNIVERSAL) == pytest.approx(21, abs=9)


def test_universal_code_with_variable_names():
    message = "table_name = ast.table_name"  # http://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=19
    assert isinstance(estimate_text_tokens(message, TextEstimator.RULE_BASED.UNIVERSAL), int)
    assert estimate_text_tokens(message, TextEstimator.RULE_BASED.UNIVERSAL) == pytest.approx(9, abs=3)


def test_universal_text_empty_and_whitespace():
    message1 = ""
    assert isinstance(estimate_text_tokens(message1, TextEstimator.RULE_BASED.UNIVERSAL), int)
    assert estimate_text_tokens(message1, TextEstimator.RULE_BASED.UNIVERSAL) == 0

    message2 = " \t \n "
    assert isinstance(estimate_text_tokens(message2, TextEstimator.RULE_BASED.UNIVERSAL), int)
    assert estimate_text_tokens(message2, TextEstimator.RULE_BASED.UNIVERSAL) == pytest.approx(5, abs=25)


def test_universal_text_non_english_with_special_chars():
    message = "versión británica"  # https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=13
    assert isinstance(estimate_text_tokens(message, TextEstimator.RULE_BASED.UNIVERSAL), int)
    assert estimate_text_tokens(message, TextEstimator.RULE_BASED.UNIVERSAL) == pytest.approx(7, abs=2)


def test_universal_text_non_english():
    message = "如何在sd上无错误进行模型训练"  # https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=20
    assert isinstance(estimate_text_tokens(message, TextEstimator.RULE_BASED.UNIVERSAL), int)
    assert estimate_text_tokens(message, TextEstimator.RULE_BASED.UNIVERSAL) == pytest.approx(31, abs=19)


def test_openai_GPT_3_5_text_with_simple_prompt():
    message = "You are the text completion model"  # https://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=2
    assert isinstance(estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_3_5), int)
    assert estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_3_5) == pytest.approx(6, abs=6)


def test_openai_GPT_3_5_text_with_punctuation():
    # https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=4
    message = "Hey there! Are you familiar with reality shifting?"
    assert isinstance(estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_3_5), int)
    assert estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_3_5) == pytest.approx(10, abs=8)


def test_openai_GPT_3_5_text_with_code_keywords():
    # https://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=13
    message = "if i ask in ten minutes will you still remember"
    assert isinstance(estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_3_5), int)
    assert estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_3_5) == pytest.approx(10, abs=7)


def test_openai_GPT_3_5_text_with_long_word():
    message = "This is a verylongwordwithoutspaces and should be counted properly."
    assert isinstance(estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_3_5), int)
    assert estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_3_5) == pytest.approx(15, abs=11)


def test_openai_GPT_3_5_text_with_url():
    # https://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=29
    message = "Analizza il contenuto di questo link https://www.deklasrl.com/siti-web-cosenza/"
    assert isinstance(estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_3_5), int)
    assert estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_3_5) == pytest.approx(31, abs=4)


def test_openai_GPT_3_5_text_with_rare_character():
    # https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=18
    message = "What is the smallest possible value for P[A ∩ B ∩ C]?"
    assert isinstance(estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_3_5), int)
    assert estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_3_5) == pytest.approx(18, abs=4)


def test_openai_GPT_3_5_text_with_newlines():
    message = "Line1\nLine2\nLine3"
    assert isinstance(estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_3_5), int)
    assert estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_3_5) == pytest.approx(8, abs=3)


def test_openai_GPT_3_5_text_with_numbers():
    # https://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=13
    message = "doesnt it have 56 floors and 202 rooms"
    assert isinstance(estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_3_5), int)
    assert estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_3_5) == pytest.approx(9, abs=6)


def test_openai_GPT_3_5_text_with_non_english():
    message = "如何在sd上无错误进行模型训练"  # https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=20
    assert isinstance(estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_3_5), int)
    assert estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_3_5) == pytest.approx(31, abs=19)


def test_openai_gpt_4_text_with_non_english():
    message = "如何在sd上无错误进行模型训练"  # https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=20
    assert isinstance(estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_4), int)
    assert estimate_text_tokens(message, TextEstimator.RULE_BASED.GPT_4) == pytest.approx(31, abs=19)


def test_raises_error_for_invalid_text():
    invalid_text = 12345
    with pytest.raises(ValueError, match=INVALID_TEXT_MESSAGE):
        estimate_text_tokens(invalid_text)


def test_raises_error_for_invalid_estimator():
    valid_text = "sample prompt"
    invalid_estimator = "not a valid estimator"
    with pytest.raises(ValueError, match=INVALID_TEXT_ESTIMATOR_MESSAGE):
        estimate_text_tokens(valid_text, invalid_estimator)
