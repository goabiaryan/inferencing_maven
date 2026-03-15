#!/usr/bin/env bash
# Launch Jupyter on Modal with A10 GPU. Open vllm_chat_and_crewai_demo.ipynb in the browser.
# Requires: pip install modal && modal token set
cd "$(dirname "$0")"
modal launch jupyter --gpu a10g
