import pytest
from tocount import estimate_text_tokens, TextEstimator


def test_linear_english_text_with_simple_prompt():
    message = "You are the text completion model"  # https://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=2
    assert isinstance(estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH) == pytest.approx(6, abs=2)


def test_linear_english_text_with_contractions():
    # https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=0
    message = "I’m refining a foolproof method for reality shifting"
    assert isinstance(estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH) == pytest.approx(10, abs=2)


def test_linear_english_text_with_prefixes_and_suffixes():
    # https://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=10
    message = "reflecting the hardships of the preparation process"
    assert isinstance(estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH) == pytest.approx(8, abs=4)


def test_linear_english_code_with_keywords():
    # http://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=19
    message1 = "def __init__(self, schema):"
    assert isinstance(estimate_text_tokens(message1, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message1, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH) == pytest.approx(8, abs=1)

    message2 = "class QueryPlanner:"  # http://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=19
    assert isinstance(estimate_text_tokens(message2, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message2, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH) == pytest.approx(5, abs=0)

    message3 = """
    for op in operations:
        if op.type == "SELECT":
    """  # http://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=19
    assert isinstance(estimate_text_tokens(message3, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message3, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH) == pytest.approx(16, abs=2)


def test_linear_english_code_with_variable_names():
    message = "table_name = ast.table_name"  # http://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=19
    assert isinstance(estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH) == pytest.approx(6, abs=1)


def test_linear_english_text_empty_and_whitespace():
    message1 = ""
    assert isinstance(estimate_text_tokens(message1, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message1, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH) == pytest.approx(0, abs=1)

    message2 = " \t \n "
    assert isinstance(estimate_text_tokens(message2, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message2, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH) == pytest.approx(3, abs=1)


def test_linear_english_text_with_long_word():
    message = "This is a verylongwordwithoutspaces and should be counted properly."
    assert isinstance(estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH) == pytest.approx(14, abs=1)


def test_linear_english_text_with_rare_character():
    # https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=18
    message = "What is the smallest possible value for P[A ∩ B ∩ C]?"
    assert isinstance(estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH) == pytest.approx(16, abs=4)


def test_linear_all_text_with_simple_prompt():
    message = "You are the text completion model"  # https://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=2
    assert isinstance(estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL), int)
    assert estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL) == pytest.approx(6, abs=6)


def test_linear_all_text_with_contractions():
    # https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=0
    message = "I’m refining a foolproof method for reality shifting"
    assert isinstance(estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL), int)
    assert estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL) == pytest.approx(10, abs=6)


def test_linear_all_text_with_prefixes_and_suffixes():
    # https://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=10
    message = "reflecting the hardships of the preparation process"
    assert isinstance(estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL), int)
    assert estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL) == pytest.approx(8, abs=8)


def test_linear_all_code_with_keywords():
    # http://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=19
    message1 = "def __init__(self, schema):"
    assert isinstance(estimate_text_tokens(message1, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL), int)
    assert estimate_text_tokens(message1, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL) == pytest.approx(8, abs=3)

    message2 = "class QueryPlanner:"  # http://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=19
    assert isinstance(estimate_text_tokens(message2, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL), int)
    assert estimate_text_tokens(message2, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL) == pytest.approx(5, abs=4)

    message3 = """
    for op in operations:
        if op.type == "SELECT":
    """  # http://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=19
    assert isinstance(estimate_text_tokens(message3, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL), int)
    assert estimate_text_tokens(message3, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL) == pytest.approx(16, abs=3)


def test_linear_all_code_with_variable_names():
    message = "table_name = ast.table_name"  # http://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=19
    assert isinstance(estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL), int)
    assert estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL) == pytest.approx(6, abs=5)


def test_linear_all_text_empty_and_whitespace():
    message1 = ""
    assert isinstance(estimate_text_tokens(message1, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL), int)
    assert estimate_text_tokens(message1, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL) == pytest.approx(0, abs=5)

    message2 = " \t \n "
    assert isinstance(estimate_text_tokens(message2, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL), int)
    assert estimate_text_tokens(message2, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL) == pytest.approx(3, abs=3)


def test_linear_all_text_non_english_with_special_chars():
    message = "versión británica"  # https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=13
    assert isinstance(estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL), int)
    assert estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL) == pytest.approx(5, abs=4)


def test_linear_all_text_non_english():
    message = "如何在sd上无错误进行模型训练"  # https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=20
    assert isinstance(estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL), int)
    assert estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL) == pytest.approx(15, abs=7)


def test_linear_all_text_with_long_word():
    message = "This is a verylongwordwithoutspaces and should be counted properly."
    assert isinstance(estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL), int)
    assert estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL) == pytest.approx(14, abs=6)


def test_linear_all_text_with_rare_character():
    # https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=18
    message = "What is the smallest possible value for P[A ∩ B ∩ C]?"
    assert isinstance(estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL), int)
    assert estimate_text_tokens(message, TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL) == pytest.approx(16, abs=1)