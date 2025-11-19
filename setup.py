# -*- coding: utf-8 -*-
"""Setup module."""
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def get_requires() -> list:
    """Read requirements.txt."""
    requirements = open("requirements.txt", "r").read()
    return list(filter(lambda x: x != "", requirements.split()))


def read_description() -> str:
    """Read README.md and CHANGELOG.md."""
    try:
        with open("README.md") as r:
            description = "\n"
            description += r.read()
        with open("CHANGELOG.md") as c:
            description += "\n"
            description += c.read()
        return description
    except Exception:
        return '''ToCount is a lightweight and extensible Python library for estimating token counts from text inputs using both rule-based and machine learning methods.
        Designed for flexibility, speed, and accuracy, ToCount provides a unified interface for different estimation strategies, making it ideal for tasks like prompt analysis,
        token budgeting, and optimizing interactions with token-based systems.'''


setup(
    name='tocount',
    packages=[
        'tocount', ],
    version='0.3',
    description='ToCount: Lightweight Token Estimator',
    long_description=read_description(),
    long_description_content_type='text/markdown',
    author='ToCount Development Team',
    author_email='tocount@openscilab.com',
    url='https://github.com/openscilab/tocount',
    download_url='https://github.com/openscilab/tocount/tarball/v0.3',
    keywords="token tokenizer estimation llm ml nlp",
    project_urls={
            'Source': 'https://github.com/openscilab/tocount',
    },
    install_requires=get_requires(),
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Manufacturing',
        'Intended Audience :: Science/Research',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    license='MIT',
)
