<div align="center">
    <h1>ToCount: Lightweight Token Estimator</h1>
    <br/>
    <a href="https://badge.fury.io/py/tocount"><img src="https://badge.fury.io/py/tocount.svg" alt="PyPI version"></a>
    <a href="https://codecov.io/gh/openscilab/tocount"><img src="https://codecov.io/gh/openscilab/tocount/branch/dev/graph/badge.svg?token=T9T0EPB3V2"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/built%20with-Python3-green.svg" alt="built with Python3"></a>
    <a href="https://github.com/openscilab/tocount"><img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/openscilab/tocount"></a>
    <a href="https://discord.gg/X8ExzygDGf"><img src="https://img.shields.io/discord/1064533716615049236.svg" alt="Discord Channel"></a>
</div>

----------


## Overview
<p align="justify">
ToCount is a lightweight and extensible Python library for estimating token counts from text inputs using both rule-based and machine learning methods. Designed for flexibility, speed, and accuracy, ToCount provides a unified interface for different estimation strategies, making it ideal for tasks like prompt analysis, token budgeting, and optimizing interactions with token-based systems.
</p>

<table>
    <tr>
        <td align="center">PyPI Counter</td>
        <td align="center">
            <a href="https://pepy.tech/projects/tocount">
                <img src="https://static.pepy.tech/badge/tocount">
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">Github Stars</td>
        <td align="center">
            <a href="https://github.com/openscilab/tocount">
                <img src="https://img.shields.io/github/stars/openscilab/tocount.svg?style=social&label=Stars">
            </a>
        </td>
    </tr>
</table>
<table>
    <tr> 
        <td align="center">Branch</td>
        <td align="center">main</td>
        <td align="center">dev</td>
    </tr>
    <tr>
        <td align="center">CI</td>
        <td align="center">
            <img src="https://github.com/openscilab/tocount/actions/workflows/test.yml/badge.svg?branch=main">
        </td>
        <td align="center">
            <img src="https://github.com/openscilab/tocount/actions/workflows/test.yml/badge.svg?branch=dev">
            </td>
    </tr>
</table>
<table>
    <tr> 
        <td align="center">Code Quality</td>
        <td align="center"><a href="https://www.codefactor.io/repository/github/openscilab/tocount"><img src="https://www.codefactor.io/repository/github/openscilab/tocount/badge" alt="CodeFactor"></a></td>
        <td align="center"><a href="https://app.codacy.com/gh/openscilab/tocount/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade"><img src="https://app.codacy.com/project/badge/Grade/4083b065eacf4587b84b557a830ee423"></a></td>
    </tr>
</table>


## Installation

### PyPI
- Check [Python Packaging User Guide](https://packaging.python.org/installing/)
- Run `pip install tocount==0.3`
### Source code
- Download [Version 0.3](https://github.com/openscilab/tocount/archive/v0.3.zip) or [Latest Source](https://github.com/openscilab/tocount/archive/dev.zip)
- Run `pip install .`

## Models

### Rule-Based


| Model Name                 |   MAE   |     MSE     |   R²   |
|----------------------------|---------|-------------|--------|
| `RULE_BASED.UNIVERSAL`     | 106.70  | 381,647.81  | 0.8175 |
| `RULE_BASED.GPT_4`         | 152.34  | 571,795.89  | 0.7266 |
| `RULE_BASED.GPT_3_5`       | 161.93  | 652,923.59  | 0.6878 |

### Tiktoken R50K

| Model Name                          |   MAE   |     MSE     |   R²   |
|-------------------------------------|---------|-------------|--------|
| `TIKTOKEN_R50K.LINEAR_ALL`          |  71.38  |  183897.01  | 0.8941 |
| `TIKTOKEN_R50K.LINEAR_ENGLISH`      |  23.35  |  14127.92   | 0.9887 |

### Tiktoken CL100K

| Model Name                          |   MAE   |     MSE     |   R²   |
|-------------------------------------|---------|-------------|--------|
| `TIKTOKEN_CL100K.LINEAR_ALL`        |  41.85  |  47949.48   | 0.9545 |
| `TIKTOKEN_CL100K.LINEAR_ENGLISH`    |  21.12  |  17597.20   | 0.9839 |

### Tiktoken O200K

| Model Name                          |   MAE   |     MSE     |   R²   |
|-------------------------------------|---------|-------------|--------|
| `TIKTOKEN_O200K.LINEAR_ALL`         |  25.53  |  20195.32   | 0.9777 |
| `TIKTOKEN_O200K.LINEAR_ENGLISH`     |  20.24  |  15887.99   | 0.9859 |


ℹ️ The training and testing dataset is taken from Lmsys-chat-1m [1] and Wildchat [2].

## Usage

```pycon
>>> from tocount import estimate_text_tokens, TextEstimator
>>> estimate_text_tokens("How are you?", estimator=TextEstimator.RULE_BASED.UNIVERSAL)
4
```

## Issues & bug reports

Just fill an issue and describe it. We'll check it ASAP! or send an email to [tocount@openscilab.com](mailto:tocount@openscilab.com "tocount@openscilab.com"). 

- Please complete the issue template

You can also join our discord server

<a href="https://discord.gg/X8ExzygDGf">
  <img src="https://img.shields.io/discord/1064533716615049236.svg?style=for-the-badge" alt="Discord Channel">
</a>

## References

<blockquote>1- Zheng, Lianmin, et al. "Lmsys-chat-1m: A large-scale real-world llm conversation dataset." International Conference on Learning Representations (ICLR) 2024 Spotlights.</blockquote>

<blockquote>2- Zhao, Wenting, et al. "Wildchat: 1m chatgpt interaction logs in the wild." International Conference on Learning Representations (ICLR) 2024 Spotlights.</blockquote>

## Show your support


### Star this repo

Give a ⭐️ if this project helped you!

### Donate to our project
If you do like our project and we hope that you do, can you please support us? Our project is not and is never going to be working for profit. We need the money just so we can continue doing what we do ;-) .			

<a href="https://openscilab.com/#donation" target="_blank"><img src="https://github.com/openscilab/tocount/raw/main/otherfiles/donation.png" width="270" alt="ToCount Donation"></a>
