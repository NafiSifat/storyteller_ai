-Software required:
Install Ollama and install the following models: 
https://ollama.com/download/windows

ollama run llama3.1:8b
ollama run deepseek-r1:7b
ollama run gemma3:4b

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

-Install Mastering Digital Image Alchemy:
https://imagemagick.org/

-Download BookCorpus:
https://www.kaggle.com/datasets/rajendrabaskota/bookcropus

-Software pipeline workthrough:
1. Data Generation -> data_gen.py
2. Fine-tuning files -> ft_gpt2_v1.py and ft_gpt2_v2.py
3. Story generation -> story_gen.py
4. Texts metrics (story metric) -> text_met.py
5. Image Generation -> Image_gen.py
6. Images metrics -> image_met.py
7. Video compilation -> movie_com.py
