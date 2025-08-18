import pytest
from tocount import universal_tokens_estimator, openai_tokens_estimator_gpt_3_5, openai_tokens_estimator_gpt_4


def test_universal_text_with_simple_prompt():
    message = "You are the text completion model"  # https://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=2
    assert isinstance(universal_tokens_estimator(message), int)
    assert universal_tokens_estimator(message) == pytest.approx(6, rel=0.8)


def test_universal_text_with_contractions():
    message = "I’m refining a foolproof method for reality shifting" # https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=0
    assert isinstance(universal_tokens_estimator(message), int)
    assert universal_tokens_estimator(message) == pytest.approx(12, rel=0.8)


def test_universal_text_with_prefixes_and_suffixes(): # https://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=10
    message = "reflecting the hardships of the preparation process"
    assert isinstance(universal_tokens_estimator(message), int)
    assert universal_tokens_estimator(message) == pytest.approx(8, rel=3)


def test_universal_code_with_keywords():
    message1 = "def __init__(self, schema):" # http://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=19
    assert isinstance(universal_tokens_estimator(message1), int)
    assert universal_tokens_estimator(message1) == pytest.approx(9, rel=0.5)

    message2 = "class QueryPlanner:" # http://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=19
    assert isinstance(universal_tokens_estimator(message2), int)
    assert universal_tokens_estimator(message2) == pytest.approx(5, rel=0.3)

    message3 = """
    for op in operations:
        if op.type == "SELECT":
    """ # http://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=19
    assert isinstance(universal_tokens_estimator(message3), int)
    assert universal_tokens_estimator(message3) == pytest.approx(21, rel=0.4)


def test_universal_code_with_variable_names():
    message = "table_name = ast.table_name" # http://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=19
    assert isinstance(universal_tokens_estimator(message), int)
    assert universal_tokens_estimator(message) == pytest.approx(9, rel=0.3)


def test_universal_text_empty_and_whitespace():
    message1 = ""
    assert isinstance(universal_tokens_estimator(message1), int)
    assert universal_tokens_estimator(message1) == 0

    message2 = " \t \n "
    assert isinstance(universal_tokens_estimator(message2), int)
    assert universal_tokens_estimator(message2) == pytest.approx(5,rel=5)


def test_universal_non_english_with_special_chars():
    message = "versión británica" #https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=13
    assert isinstance(universal_tokens_estimator(message), int)
    assert universal_tokens_estimator(message) == pytest.approx(7, rel=0.2)


def test_universal_non_english():
    message = "如何在sd上无错误进行模型训练"  # https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=20
    assert isinstance(universal_tokens_estimator(message), int)
    assert universal_tokens_estimator(message) == pytest.approx(31, rel=0.6)


def test_openai_heuristic_text_with_simple_prompt():
    message = "You are the text completion model"  # https://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=2
    assert isinstance(openai_tokens_estimator_gpt_3_5(message), int)
    assert openai_tokens_estimator_gpt_3_5(message) == pytest.approx(6, rel=0.9)


def test_openai_heuristic_text_with_punctuation():
    message = "Hey there! Are you familiar with reality shifting?" # https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=4
    assert isinstance(openai_tokens_estimator_gpt_3_5(message), int)
    assert openai_tokens_estimator_gpt_3_5(message) == pytest.approx(10, rel=0.8)


def test_openai_heuristic_text_with_code_keywords():
    message = "if i ask in ten minutes will you still remember" # https://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=13
    assert isinstance(openai_tokens_estimator_gpt_3_5(message), int)
    assert openai_tokens_estimator_gpt_3_5(message) == pytest.approx(10, rel=0.7)


def test_openai_heuristic_text_with_long_word():
    message = "This is a verylongwordwithoutspaces and should be counted properly."
    assert isinstance(openai_tokens_estimator_gpt_3_5(message), int)
    assert openai_tokens_estimator_gpt_3_5(message) == pytest.approx(15, rel=0.7)


def test_openai_heuristic_text_with_url():
    message = "Analizza il contenuto di questo link https://www.deklasrl.com/siti-web-cosenza/" # https://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=29
    assert isinstance(openai_tokens_estimator_gpt_3_5(message), int)
    assert openai_tokens_estimator_gpt_3_5(message) == pytest.approx(31, rel=0.1)


def test_openai_heuristic_text_with_rare_character():
    message = "What is the smallest possible value for P[A ∩ B ∩ C]?" # https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=18
    assert isinstance(openai_tokens_estimator_gpt_3_5(message), int)
    assert openai_tokens_estimator_gpt_3_5(message) == pytest.approx(18, rel=0.2)


def test_openai_heuristic_text_with_newlines():
    message = "Line1\nLine2\nLine3"
    assert isinstance(openai_tokens_estimator_gpt_3_5(message), int)
    assert openai_tokens_estimator_gpt_3_5(message) == pytest.approx(8, rel=0.3)


def test_openai_heuristic_text_with_numbers():
    message = "doesnt it have 56 floors and 202 rooms" # https://huggingface.co/datasets/lmsys/lmsys-chat-1m?conversation-viewer=13
    assert isinstance(openai_tokens_estimator_gpt_3_5(message), int)
    assert openai_tokens_estimator_gpt_3_5(message) == pytest.approx(9, rel=0.6)


def test_openai_gpt4_model_adjustment():
    message = "A simple sentence to compare models."
    result_3_5 = openai_tokens_estimator_gpt_3_5(message)
    result_4 = openai_tokens_estimator_gpt_4(message)

    assert isinstance(result_3_5, int)
    assert isinstance(result_4, int)
    assert result_4 >= result_3_5


def test_openai_gpt35_estimator_with_non_english():
    message = "如何在sd上无错误进行模型训练"  # https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=20
    assert isinstance(openai_tokens_estimator_gpt_3_5(message), int)
    assert openai_tokens_estimator_gpt_3_5(message) == pytest.approx(31, rel=0.6)


def test_openai_gpt4_estimator_with_non_english():
    message = "如何在sd上无错误进行模型训练"  # https://huggingface.co/datasets/allenai/WildChat-1M?conversation-viewer=20
    assert isinstance(openai_tokens_estimator_gpt_4(message), int)
    assert openai_tokens_estimator_gpt_4(message) == pytest.approx(31, rel=0.6)