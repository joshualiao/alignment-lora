CS182

Installation Instructions:

see https://github.com/huggingface/diffusers/tree/main/examples/text_to_image LoRA section

git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .

pip install -r requirements.txt
accelerate config
# config accelerate with fp16

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

huggingface-cli login
