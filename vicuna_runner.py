import sys
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Function to generate responses from input strings
def generate_responses(input_csv, output_csv, model_id, max_length, suffix_string):
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    df = pd.read_csv(input_csv)
    df = df.values
    questions = [str(prompt) for prompt in df.T[0]]
    responses = []

    for input_text in tqdm(questions):
        input_ids = tokenizer(input_text+" "+suffix_string, return_tensors="pt").input_ids
        input_ids = input_ids.to("cuda")
        output = model.generate(input_ids, max_length=max_length)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        responses.append([input_text, output_text])

    response_df = pd.DataFrame(responses, columns=["question", "response"])
    response_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_responses.py input.csv max_length suffix_string")
        sys.exit(1)

    input_csv = sys.argv[1]
    max_length = int(sys.argv[2])
    suffix_string = sys.argv[3]
    output_csv = "latest_response.csv"
    model_id = "lmsys/vicuna-7b-v1.5"

    generate_responses(input_csv, output_csv, model_id, max_length, suffix_string)
