import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def generate_story(base_model_path, adapter_path, input_text):
    """
    Loads a fine-tuned model and generates a story based on input text.
    """
    print(f"Loading base model from local path: {base_model_path}")
    # Load the base model in 4-bit for efficient inference
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, # Load from local directory
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path) # Load from local directory
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the fine-tuned adapter
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # Format the prompt exactly as it was during training
    # NOTE: This prompt template must match what was used in the training data
    instruction = "Based on the participant's uP! Life Report, generate an engaging, shareable story-based report about them."
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\n--- uP! Life Report Input ---\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    # Generate the story
    print("\n--- Generating Story ---")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=1500, 
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response
    story = response.split("<|end_header_id|>")[-1].strip()

    print("\n--- Generated Report ---")
    print(story)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a story from a fine-tuned model.")
    parser.add_argument("--base_model_path", type=str, required=True, help="The local directory path to the base model.")
    parser.add_argument("--adapter_path", type=str, required=True, help="The path to the saved LoRA adapter directory.")
    parser.add_argument("--input_text", type=str, required=True, help="The input text from a new uP! Life Report.")
    
    args = parser.parse_args()
    generate_story(args.base_model_path, args.adapter_path, args.input_text)
