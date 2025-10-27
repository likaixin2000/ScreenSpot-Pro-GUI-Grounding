import os
import re

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, GenerationConfig



# Second round chat 
# pass history context of multi-turn conversation
# msgs.append({"role": "assistant", "content": [answer]})
# msgs.append({"role": "user", "content": ["Introduce something about Airbus A380."]})

# answer = model.chat(
#     image=None,
#     msgs=msgs,
#     tokenizer=tokenizer
# )
# print(answer)

class MiniCPMVModel():
    def __init__(self):
        pass

    def load_model(self, model_name_or_path="openbmb/MiniCPM-V-2_6", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device).eval()

        # Setting default generation config
        self.override_generation_config = GenerationConfig.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            max_length=None,
        )
    
    def set_generation_config(self, **kwargs):
        self.override_generation_config.update(kwargs)
        self.model.generation_config = GenerationConfig(**self.override_generation_config)

    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."
        
        # Prepare query
        grounding_prompt = f'What is the bounding box of the UI element corresponding to the user instruction "{instruction}"? Output in the format of x1 x2 y1 y2.'
        
        msgs = [{'role': 'user', 'content': [image, grounding_prompt]}]

        response = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )
        # print(response)
        # Extract bounding box
        # print("------")
        # print(grounding_prompt)
        # print("------")
        # print(response)
        # print("------")
        # Try getting groundings
        bbox = extract_first_bounding_box(response)
        if bbox:
            click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        else:
            # Try matching a point
            click_point = extract_first_point(response)

        result_dict = {
            "result": "positive",
            "bbox": bbox,
            "point": click_point,
            "raw_response": response
        }
        
        return result_dict


def extract_first_bounding_box(text):
    # Regular expression pattern to match the first bounding box in the format <box>x0 y0 x1 y1</box>
    pattern = r"<box>(\d+) (\d+) (\d+) (\d+)</box>"
    
    # Search for the first match in the text with the DOTALL flag to support multi-line text
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        bbox = [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
        return [pos / 1000 for pos in bbox]
    
    return None


def extract_first_point(text):
    # Regular expression pattern to match the first point
    pattern = r"(\d+) (\d+)"
    
    # Search for the first match in the text with the DOTALL flag to support multi-line text
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        bbox = [int(match.group(1)), int(match.group(2))]
        return [pos / 1000 for pos in bbox]
    
    return None
