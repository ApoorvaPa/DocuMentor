# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import torch

# def load_llm(model_name="TinyLlama/TinyLlama-1.1B", quantized=False):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#         device_map="auto"
#     )

#     pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
#     return pipe
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def load_llm(model_name="tiiuae/falcon-rw-1b"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    return pipe
