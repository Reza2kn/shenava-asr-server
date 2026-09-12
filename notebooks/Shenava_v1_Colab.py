# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   colab:
#     name: Shenava v1 - Persian ASR.ipynb
# ---

# %% [markdown]
# # Shenava v1 — Persian speech recognition
#
# Run **Setup** once, then run exactly one of the three model boxes. Each box downloads the
# matching model and lets you upload an audio file. No GPU is required.
#
# - **Koochik (114M):** best accuracy, largest download (~459 MB)
# - **Rizeh (32M):** balanced (~117 MB)
# - **Rizeh-Pizeh (6.9M):** smallest and fastest (~33 MB)
#
# The optional **3,669-word hotbeam** result uses Shenava's fixed Persian hotword list,
# beam width 80, and hotword weight 2.5. It is useful for names and domain terms; greedy can
# still be better on some general speech, so both outputs are shown.

# %% [markdown]
# ## 1. Setup — run once

# %%
%pip install -q "sherpa-onnx==1.13.8" "onnxruntime==1.23.2" "huggingface-hub==0.35.3" "soundfile==0.13.1"

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ORT_NUM_THREADS"] = "1"

!wget -q -O /content/shenava_colab.py https://raw.githubusercontent.com/Reza2kn/shenava-asr-server/main/notebooks/shenava_colab.py

from shenava_colab import upload_and_transcribe
print("✓ Shenava is ready. Run one model box below.")

# %% [markdown]
# ## 2A. Koochik v1.0 — best accuracy + 3,669-word hotbeam
#
# Run this cell, choose an audio file, and wait for both the normal greedy transcript and
# Shenava hotbeam transcript.

# %%
upload_and_transcribe("koochik", include_hotbeam=True)

# %% [markdown]
# ## 2B. Rizeh v1.0 — balanced

# %%
upload_and_transcribe("rizeh")

# %% [markdown]
# ## 2C. Rizeh-Pizeh v1.0 — smallest and fastest

# %%
upload_and_transcribe("rizeh-pizeh")

# %% [markdown]
# ## Troubleshooting
#
# - Run **Setup** before a model box.
# - Do not install another sherpa-onnx, onnxruntime, or pyctcdecode version afterward.
# - The helper keeps only one sherpa recognizer alive at a time; switching boxes releases the
#   previous one before loading the next.
# - Audio is mixed to mono and sherpa resamples it to 16 kHz automatically.
# - If a download was interrupted, rerun the same model box; Hugging Face resumes its cache.

# %% [markdown]
# ## 2A. Koochik v1.0 — 114M
# Best accuracy. Shows both greedy and the 3,669-word hotbeam result.

# %%
upload_and_transcribe("koochik", include_hotbeam=True)

# %% [markdown]
# ## 2B. Rizeh v1.0 — 32M
# Balanced size and accuracy. Shows both greedy and the 3,669-word hotbeam result.

# %%
upload_and_transcribe("rizeh", include_hotbeam=True)

# %% [markdown]
# ## 2C. Rizeh-Pizeh v1.0 — 6.9M
# Tiny and fast. Shows both greedy and the 3,669-word hotbeam result.

# %%
upload_and_transcribe("rizeh-pizeh", include_hotbeam=True)

# %% [markdown]
# ## Troubleshooting
#
# If Colab has already imported another `sherpa_onnx` or `onnxruntime` build, choose
# **Runtime → Restart session**, run Setup once, then run one model box. The helper intentionally
# keeps only one native recognizer alive at a time to avoid memory spikes.
