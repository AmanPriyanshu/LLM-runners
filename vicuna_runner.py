import sys
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Function to generate responses from input strings
def generate_responses(input_csv, output_csv, model_id, device):
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.to(device)

    df = pd.read_csv(input_csv)
    df = df.values
    questions = [str(prompt) for prompt in df.T[0]]
    responses = []

    for input_text in tqdm(questions):
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        output = model.generate(input_ids, max_length=50)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        responses.append([input_text, output_text])

    response_df = pd.DataFrame(responses, columns=["question", "response"])
    response_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_responses.py input.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = "latest_response.csv"
    model_id = "lmsys/vicuna-7b-v1.5"
    device = 0  # Set your desired GPU device here

    generate_responses(input_csv, output_csv, model_id, device)
