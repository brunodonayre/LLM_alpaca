!pip3 install autotrain-advanced
!autotrain setup --update-torch
# Connect to my google drive to record the training data

from google.colab import drive
drive.mount('/content/drive')

!autotrain llm --train --project-name "alpaca7B" --model "4i-ai/Llama-2-7b-alpaca-es" --data-path "/content/drive/MyDrive/NLP/sesion11/Datos" --batch-size 2 --epochs 2 --trainer sft  --quantization "int4" --lora-r 16 --lora-alpha 32 --save_total_limit 1

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model (replace with your actual model name if different)
model = AutoModelForCausalLM.from_pretrained("4i-ai/Llama-2-7b-alpaca-es")
tokenizer = AutoTokenizer.from_pretrained("4i-ai/Llama-2-7b-alpaca-es")

# Save to a new directory
save_path = "/content/drive/MyDrive/NLP/sesion11/Modelos/alpaca7B_saved"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# Verify files
!ls -l /content/drive/MyDrive/NLP/sesion11/Modelos/alpaca7B_saved

# Verify files
!ls -l /content/drive/MyDrive/NLP/sesion11/Modelos/alpaca7B_saved

import torch

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Check CUDA version
print("CUDA version:", torch.version.cuda)

# Check GPU model (e.g., T4, A100)
print("GPU:", torch.cuda.get_device_name(0))

!pip install -U bitsandbytes pip triton transformers accelerate

##Cargar el modelo realizado el finetunning
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
model_name = "/content/drive/MyDrive/NLP/sesion11/Modelos/alpaca7B_saved"

# Clear GPU cache first
torch.cuda.empty_cache()

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=True)

    def create_and_prepare_model():
            compute_dtype = getattr(torch, "float16")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,  torch_dtype=compute_dtype, local_files_only=True
            )
            return model
    model = create_and_prepare_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")

from transformers import GenerationConfig

def generate(instruction, input=None):
    #Format the prompt to look like the training data
    if input is not None:
        prompt = "### Instruction:\n"+instruction+"\n\n### Input:\n"+input+"\n\n### Response:\n"
    else :
        prompt = "### Instruction:\n"+instruction+"\n\n### Response:\n"


    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    generation_output = model.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(temperature=1.0, top_p=0.75, top_k=40, num_beams=10), #hyperparameters for generation
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=150, #maximum tokens generated, increase if you want longer asnwer (up to 2048 - the length of the prompt), generation "looks" slower for longer response

    )
    for seq in generation_output.sequences:
        output = tokenizer.decode(seq, skip_special_tokens=True)
        return output.split("### Response:")[1].strip()


from transformers import GenerationConfig

print(generate("¿Qué es la Ley de la Salud en el Perú?"))
print("--------------------------------")
print(generate("¿Qué funcion tiene el comité farmaceutico del perú?"))
print("--------------------------------")
print(generate("¿Qué significa PNUME?"))

!pip uninstall torchvision -y
!pip install torchvision --force-reinstall

#####################

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_name_hub = "4i-ai/Llama-2-7b-alpaca-es"

# Clear GPU cache first
torch.cuda.empty_cache()

tokenizer_hub = AutoTokenizer.from_pretrained(model_name_hub, use_fast=True)

def create_and_prepare_model_hub():
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model_hub = AutoModelForCausalLM.from_pretrained(
        model_name_hub,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype
    )
    return model_hub

try:
    model_hub = create_and_prepare_model_hub()
    print("Model loaded successfully from Hugging Face Hub!")
except Exception as e:
    print(f"Error loading model from Hugging Face Hub: {str(e)}")

############################################################
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_hub = "4i-ai/Llama-2-7b-alpaca-es"

# Clear GPU cache first
torch.cuda.empty_cache()

tokenizer_hub = AutoTokenizer.from_pretrained(model_name_hub, use_fast=True)

def create_and_prepare_model_hub():
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model_hub = AutoModelForCausalLM.from_pretrained(
        model_name_hub,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype
    )
    return model_hub

try:
    model_hub = create_and_prepare_model_hub()
    print("Model loaded successfully from Hugging Face Hub!")
except Exception as e:
    print(f"Error loading model from Hugging Face Hub: {str(e)}")


##The model has been successfully loaded from the Hugging Face Hub. You can now use it for text generation.
def generate_with_hub_model(instruction, input=None):
    #Format the prompt to look like the training data
    if input is not None:
        prompt = "### Instruction:\n"+instruction+"\n\n### Input:\n"+input+"\n\n### Response:\n"
    else :
        prompt = "### Instruction:\n"+instruction+"\n\n### Response:\n"


    inputs = tokenizer_hub(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    generation_output = model_hub.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(temperature=1.0, top_p=0.75, top_k=40, num_beams=10), #hyperparameters for generation
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=150, #maximum tokens generated, increase if you want longer asnwer (up to 2048 - the length of the prompt), generation "looks" slower for longer response

    )
    for seq in generation_output.sequences:
        output = tokenizer_hub.decode(seq, skip_special_tokens=True)
        # Assuming the model outputs in a similar format, extract the response part
        if "### Response:" in output:
            return output.split("### Response:")[1].strip()
        else:
            return output.strip() # Return the whole output if no specific response tag

print(generate_with_hub_model("¿Qué es la Ley de la Salud en el Perú?"))
print("--------------------------------")
print(generate_with_hub_model("¿Qué funcion tiene el comité farmaceutico del perú?"))
print("--------------------------------")
print(generate_with_hub_model("¿Qué significa PNUME?"))
