from setuptools import setup, find_packages

setup(
    name="gymnasium_stag_hunt",
    version="0.0.1",
    author="Giorgio Franceschelli - fork from David Nesterov-Rappoport",
    author_email="giorgio.franceschelli@unibo.it",
    description="Markov stag hunt environment for gymnasium",
    long_description="This package is based on gymnasium and a fork from the original OpenAI gym-based stag-hunt "
    "environment.",
    long_description_content_type="text/markdown",
    url="https://github.com/giorgiofranceschelli/Gymnasium-Stag-Hunt",
    packages=find_packages(),
    include_package_data=True,
    package_data={'gymnasium_stag_hunt': ['assets/*', 'assets/**/*']},
    install_requires=[
        # --- Stag Hunt env (original) ---
        "gymnasium",
        "pygame",
        "opencv-python",
        "pettingzoo",

        # --- Data & utils ---
        "numpy",
        "pandas",

        # --- LLM (frozen inference via Ollama, for qwen4b.py) ---
        "ollama",

        # --- LLM (HuggingFace, for llm_policy_agent.py LLMEncoder) ---
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "bitsandbytes>=0.43.0",     # 4-bit quantization (QLoRA)

        # --- RL training ---
        "torch>=2.2.0",

        # --- Progress bars ---
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",   # bumped from 3.9: project uses str | None syntax
)
