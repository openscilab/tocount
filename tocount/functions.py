# -*- coding: utf-8 -*-
"""Tocount functions."""
from .params import INVALID_TEXT_MESSAGE, INVALID_TEXT_ESTIMATOR_MESSAGE
from .params import TextEstimator, _TextEstimatorRuleBased, _TextEstimatorTikTokenR50K
from .params import _TextEstimatorTikTokenCL100K, _TextEstimatorTikTokenO200K
from .params import _TextEstimatorDeepseekR1, _TextEstimatorQwenQwQ, _TextEstimatorLlama_3_1
from .rule_based.functions import universal_tokens_estimator, openai_tokens_estimator_gpt_3_5, openai_tokens_estimator_gpt_4
from .tiktoken_r50k.functions import linear_tokens_estimator_all as r50k_linear_all
from .tiktoken_r50k.functions import linear_tokens_estimator_english as r50k_linear_english
from .tiktoken_cl100k.functions import linear_tokens_estimator_all as cl100k_linear_all
from .tiktoken_cl100k.functions import linear_tokens_estimator_english as cl100k_linear_english
from .tiktoken_o200k.functions import linear_tokens_estimator_all as o200k_linear_all
from .tiktoken_o200k.functions import linear_tokens_estimator_english as o200k_linear_english
from .deepseek_r1.functions import linear_tokens_estimator_all as deepseek_r1_linear_all
from .deepseek_r1.functions import linear_tokens_estimator_english as deepseek_r1_linear_english
from .qwen_qwq.functions import linear_tokens_estimator_all as qwen_qwq_linear_all
from .qwen_qwq.functions import linear_tokens_estimator_english as qwen_qwq_linear_english
from .llama_3_1.functions import linear_tokens_estimator_all as llama_3_1_linear_all
from .llama_3_1.functions import linear_tokens_estimator_english as llama_3_1_linear_english


text_estimator_map = {
    TextEstimator.RULE_BASED.UNIVERSAL: universal_tokens_estimator,
    TextEstimator.RULE_BASED.GPT_3_5: openai_tokens_estimator_gpt_3_5,
    TextEstimator.RULE_BASED.GPT_4: openai_tokens_estimator_gpt_4,
    TextEstimator.TIKTOKEN_R50K.LINEAR_ALL: r50k_linear_all,
    TextEstimator.TIKTOKEN_R50K.LINEAR_ENGLISH: r50k_linear_english,
    TextEstimator.TIKTOKEN_CL100K.LINEAR_ALL: cl100k_linear_all,
    TextEstimator.TIKTOKEN_CL100K.LINEAR_ENGLISH: cl100k_linear_english,
    TextEstimator.TIKTOKEN_O200K.LINEAR_ALL: o200k_linear_all,
    TextEstimator.TIKTOKEN_O200K.LINEAR_ENGLISH: o200k_linear_english,
    TextEstimator.DEEPSEEK_R1.LINEAR_ALL: deepseek_r1_linear_all,
    TextEstimator.DEEPSEEK_R1.LINEAR_ENGLISH: deepseek_r1_linear_english,
    TextEstimator.QWEN_QWQ.LINEAR_ALL: qwen_qwq_linear_all,
    TextEstimator.QWEN_QWQ.LINEAR_ENGLISH: qwen_qwq_linear_english,
    TextEstimator.LLAMA_3_1.LINEAR_ALL: llama_3_1_linear_all,
    TextEstimator.LLAMA_3_1.LINEAR_ENGLISH: llama_3_1_linear_english,
}


def estimate_text_tokens(text: str, estimator: TextEstimator = TextEstimator.DEFAULT) -> int:
    """
    Estimate text tokens number.

    :param text: input text
    :param estimator: estimator type
    :return: tokens number
    """
    if not isinstance(text, str):
        raise ValueError(INVALID_TEXT_MESSAGE)
    if not isinstance(estimator, (TextEstimator, _TextEstimatorRuleBased, _TextEstimatorTikTokenR50K, _TextEstimatorTikTokenCL100K, _TextEstimatorTikTokenO200K, _TextEstimatorDeepseekR1, _TextEstimatorQwenQwQ, _TextEstimatorLlama_3_1)):
        raise ValueError(INVALID_TEXT_ESTIMATOR_MESSAGE)
    return text_estimator_map[estimator](text)
