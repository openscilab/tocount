# -*- coding: utf-8 -*-
"""Tests for the Deepseek R1 token estimator."""

import pytest
from tocount import estimate_text_tokens, TextEstimator


def test_linear_english_text_with_simple_prompt():
    # https://huggingface.co/datasets/OpenAssistant/oasst1/viewer?row=2
    message = "Now explain it to a dog"
    assert isinstance(estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH) == pytest.approx(7, abs=1)


def test_linear_english_text_with_contractions():
    # https://huggingface.co/datasets/OpenAssistant/oasst1/viewer?row=32
    message = "Can you clarify the analogy? I'm not following the notation or vocabulary used in the example."
    assert isinstance(estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH) == pytest.approx(20, abs=3)


def test_linear_english_text_with_prefixes_and_suffixes():
    # https://huggingface.co/datasets/OpenAssistant/oasst1/viewer/default/train?p=2&row=277
    message = "it's always a good idea to consult an eye doctor to rule out any underlying medical conditions or eye diseases."
    assert isinstance(estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH) == pytest.approx(23, abs=4)


def test_linear_english_code_with_keywords():
    # https://huggingface.co/datasets/OpenAssistant/oasst1/viewer/default/train?p=8&row=837
    message1 = "elif a[1] == str(3) or a[1] == str(4)"
    assert isinstance(estimate_text_tokens(message1, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message1, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH) == pytest.approx(21, abs=10)

    # https://huggingface.co/datasets/OpenAssistant/oasst1/viewer/default/train?p=22&row=2237
    message2 = "Mat img = x_train[0];"
    assert isinstance(estimate_text_tokens(message2, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message2, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH) == pytest.approx(9, abs=1)

    # https://huggingface.co/datasets/OpenAssistant/oasst1/viewer/default/train?p=16&row=1624
    message3 = """def fibonacci(n):
a = 0
b = 1
for k in range(n):
c = b+a
a = b
b = c
return a"""
    assert isinstance(estimate_text_tokens(message3, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message3, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH) == pytest.approx(36, abs=16)


def test_linear_english_code_with_variable_names():
    # https://huggingface.co/datasets/OpenAssistant/oasst1/viewer/default/train?p=12&row=66
    message = "shift = request_body.get('shift', 1)"
    assert isinstance(estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH) == pytest.approx(12, abs=1)


def test_linear_english_text_empty_and_whitespace():
    message1 = ""
    assert isinstance(estimate_text_tokens(message1, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message1, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH) == pytest.approx(1, abs=2)

    message2 = " \t \n "
    assert isinstance(estimate_text_tokens(message2, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message2, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH) == pytest.approx(4, abs=0)


def test_linear_english_text_with_long_word():
    message = "This is a verylongwordwithoutspaces and should be counted properly."
    assert isinstance(estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH) == pytest.approx(16, abs=2)


def test_linear_english_text_with_rare_character():
    message = "What does the symbol Ω (Omega) represent in physics?"
    assert isinstance(estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH), int)
    assert estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH) == pytest.approx(13, abs=1)


def test_linear_all_text_with_simple_prompt():
    # https://huggingface.co/datasets/OpenAssistant/oasst1/viewer?row=2
    message = "Now explain it to a dog"
    assert isinstance(estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ALL), int)
    assert estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ALL) == pytest.approx(7, abs=7)


def test_linear_all_text_with_contractions():
    # https://huggingface.co/datasets/OpenAssistant/oasst1/viewer?row=32
    message = "Can you clarify the analogy? I'm not following the notation or vocabulary used in the example."
    assert isinstance(estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ALL), int)
    assert estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ALL) == pytest.approx(20, abs=10)


def test_linear_all_text_with_prefixes_and_suffixes():
    # https://huggingface.co/datasets/OpenAssistant/oasst1/viewer/default/train?p=2&row=277
    message = "it's always a good idea to consult an eye doctor to rule out any underlying medical conditions or eye diseases."
    assert isinstance(estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ALL), int)
    assert estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ALL) == pytest.approx(23, abs=11)


def test_linear_all_code_with_keywords():
    # https://huggingface.co/datasets/OpenAssistant/oasst1/viewer/default/train?p=8&row=837
    message1 = "elif a[1] == str(3) or a[1] == str(4)"
    assert isinstance(estimate_text_tokens(message1, TextEstimator.DEEPSEEK_R1.LINEAR_ALL), int)
    assert estimate_text_tokens(message1, TextEstimator.DEEPSEEK_R1.LINEAR_ALL) == pytest.approx(21, abs=4)

    # https://huggingface.co/datasets/OpenAssistant/oasst1/viewer/default/train?p=22&row=2237
    message2 = "Mat img = x_train[0];"
    assert isinstance(estimate_text_tokens(message2, TextEstimator.DEEPSEEK_R1.LINEAR_ALL), int)
    assert estimate_text_tokens(message2, TextEstimator.DEEPSEEK_R1.LINEAR_ALL) == pytest.approx(9, abs=5)

    # https://huggingface.co/datasets/OpenAssistant/oasst1/viewer/default/train?p=16&row=1624
    message3 = """def fibonacci(n):
a = 0
b = 1
for k in range(n):
c = b+a
a = b
b = c
return a"""
    assert isinstance(estimate_text_tokens(message3, TextEstimator.DEEPSEEK_R1.LINEAR_ALL), int)
    assert estimate_text_tokens(message3, TextEstimator.DEEPSEEK_R1.LINEAR_ALL) == pytest.approx(36, abs=10)


def test_linear_all_code_with_variable_names():
    # https://huggingface.co/datasets/OpenAssistant/oasst1/viewer/default/train?p=12&row=66
    message = "shift = request_body.get('shift', 1)"
    assert isinstance(estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ALL), int)
    assert estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ALL) == pytest.approx(12, abs=5)


def test_linear_all_text_empty_and_whitespace():
    message1 = ""
    assert isinstance(estimate_text_tokens(message1, TextEstimator.DEEPSEEK_R1.LINEAR_ALL), int)
    assert estimate_text_tokens(message1, TextEstimator.DEEPSEEK_R1.LINEAR_ALL) == pytest.approx(1, abs=8)

    message2 = " \t \n "
    assert isinstance(estimate_text_tokens(message2, TextEstimator.DEEPSEEK_R1.LINEAR_ALL), int)
    assert estimate_text_tokens(message2, TextEstimator.DEEPSEEK_R1.LINEAR_ALL) == pytest.approx(4, abs=6)


def test_linear_all_text_non_english_with_special_chars():
    # https://huggingface.co/datasets/OpenAssistant/oasst1/viewer/default/train?p=37&row=3705
    message = "¡Claro! Aquí te dejo algunas sugerencias para ambas situaciones:"
    assert isinstance(estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ALL), int)
    assert estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ALL) == pytest.approx(17, abs=6)


def test_linear_all_text_non_english():
    # https://huggingface.co/datasets/OpenAssistant/oasst1/viewer/default/train?p=37&row=3721
    message = "Es verdadera o es falsa"
    assert isinstance(estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ALL), int)
    assert estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ALL) == pytest.approx(8, abs=6)


def test_linear_all_text_with_long_word():
    message = "This is a verylongwordwithoutspaces and should be counted properly."
    assert isinstance(estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ALL), int)
    assert estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ALL) == pytest.approx(16, abs=8)


def test_linear_all_text_with_rare_character():
    message = "What does the symbol Ω (Omega) represent in physics?"
    assert isinstance(estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ALL), int)
    assert estimate_text_tokens(message, TextEstimator.DEEPSEEK_R1.LINEAR_ALL) == pytest.approx(13, abs=7)