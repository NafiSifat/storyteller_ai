NOTE: Generated files are uploaded alongside with the code for convenience. All gen folder are full with the files which used in the project report. Also the GPT-2 fine-tuned weights are not given since they are enormous (2-3GB) for github.

-----------------------------------------------------------------
-Software required:
Install Ollama and install the following models: 

https://ollama.com/download/windows

In command prompt:

ollama run llama3.1:8b

ollama run deepseek-r1:7b

ollama run gemma3:4b

-----------------------------------------------------------------
-Conda python environment libraries:

numpy

pandas

torch

torchvision

transformers

HuggingFace 

nltk

diffusers

PIL

scipy

moviepy

tqdm

matplotlib

pathlib

json

gtts

-----------------------------------------------------------------

-Install Mastering Digital Image Alchemy:
https://imagemagick.org/

-----------------------------------------------------------------

-Download BookCorpus:
https://www.kaggle.com/datasets/rajendrabaskota/bookcropus

-----------------------------------------------------------------

-Software pipeline workthrough/sequence:
1. Data Generation -> data_gen.py
2. Fine-tuning files -> ft_gpt2_v1.py and ft_gpt2_v2.py
3. Story generation -> story_gen.py
4. Texts metrics (story metric) -> text_met.py
5. Image Generation -> Image_gen.py
6. Images metrics -> image_met.py
7. Video compilation -> movie_com.py

-API call file:
1. Ollama Llama 3.1 API: llama_api.py
2. Ollama DeepSeek-R1 API: deepseek_api.py
3. Ollama Gemma 3 API: gemma3_api.py

-----------------------------------------------------------------
