"""
Agentic RAG Implementations — Setup Configuration
Paper: arXiv:2501.09136
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="agentic-rag-implementations",
    version="1.0.0",
    author="Agentic RAG Survey Reproduction",
    description=(
        "Complete implementation of all architectures from "
        "'Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG' "
        "(arXiv:2501.09136)"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asinghcsu/AgenticRAG-Survey",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "agentic-rag=run_all:main",
        ]
    },
)
