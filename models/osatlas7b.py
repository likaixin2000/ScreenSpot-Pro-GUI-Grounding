import json
import os
import re
import tempfile
import base64
from io import BytesIO
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import torch

import openai
from qwen_vl_utils import process_vision_info


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# bbox -> point (str)
def bbox_2_point(bbox, dig=2):
    # bbox [left, top, right, bottom]
    point = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    point = [f"{item:.2f}" for item in point]
    point_str = "({},{})".format(point[0], point[1])
    return point_str

# bbox -> bbox (str)
def bbox_2_bbox(bbox, dig=2):
    bbox = [f"{item:.2f}" for item in bbox]
    bbox_str = "({},{},{},{})".format(bbox[0], bbox[1], bbox[2], bbox[3])
    return bbox_str

# point (str) -> point
def pred_2_point(s):
    floats = re.findall(r'-?\d+\.?\d*', s)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        return floats
    elif len(floats) == 4:
        return [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
    else:
        return None

# bbox (qwen str) -> bbox
import re

def extract_bbox(s):
    # Extract text between <|box_start|> and <|box_end|> tags
    match = re.search(r'<\|box_start\|\>(.*?)<\|box_end\|\>', s)
    
    if match:
        # Get the text between the tags
        extracted_text = match.group(1)
        
        # Remove parentheses and brackets
        cleaned_text = re.sub(r'[()\[\]]', '', extracted_text)
        
        # Extract four numbers from the cleaned text
        pattern = r"(\d+),\s*(\d+),\s*(\d+),\s*(\d+)"
        numbers = re.findall(pattern, cleaned_text)
        
        if numbers:
            # Return the first match as tuples of integers
            x1, y1, x2, y2 = numbers[0]
            return (int(x1), int(y1)), (int(x2), int(y2))
    
    return None



def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name


class OSAtlas7BModel():
    def load_model(self, model_name_or_path="OS-Copilot/OS-Atlas-Base-7B", device="cuda"):
        self.device = device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            device_map=device, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        # Setting default generation config
        self.generation_config = GenerationConfig.from_pretrained("OS-Copilot/OS-Atlas-Base-7B", trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_length=4096,
            do_sample=False,
            temperature=0.0
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)


    def ground_only_positive(self, instruction, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."



        prompt_origin = 'In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with bbox)?"'
        full_prompt = prompt_origin.format(instruction)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]
        # Preparation for inference
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]


        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        if '<|box_start|>' in response and '<|box_end|>' in response:
            pred_bbox = extract_bbox(response)
            if pred_bbox is not None:
                (x1, y1), (x2, y2) = pred_bbox
                pred_bbox = [pos / 1000 for pos in [x1, y1, x2, y2]]
                x1, y1, x2, y2 = pred_bbox
                if 0 < x1 < 1 and 0 < y1 < 1 and 0 < x2 < 1 and 0 < y2 < 1:
                    click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
                    
                    result_dict["bbox"] = pred_bbox
                    result_dict["point"] = click_point
        else:
            print('---------------')
            print(response)
            click_point = pred_2_point(response)
            if click_point is not None:
                x, y = click_point
                if 0 < x < 1 and 0 < y < 1:
                    click_point = [x / 1000, y / 1000]
                    result_dict["point"] = click_point  # can be none
        
        return result_dict


    def ground_allow_negative(self, instruction, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        prompt_origin = 'Output the bounding box in the image corresponding to the instruction "{}". If the target does not exist, respond with "Target does not exist".'
        full_prompt = prompt_origin.format(instruction)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]
        # Preparation for inference
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]



        result_dict = {
            "result": None,
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        if '<|box_start|>' in response and '<|box_end|>' in response:
            pred_bbox = extract_bbox(response)
            if pred_bbox is not None:
                (x1, y1), (x2, y2) = pred_bbox
                pred_bbox = [pos / 1000 for pos in [x1, y1, x2, y2]]
                click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]

                result_dict["bbox"] = pred_bbox
                result_dict["point"] = click_point
        else:
            print('---------------')
            print(response)
            click_point = pred_2_point(response)
            click_point = [x / 1000 for x in click_point] if click_point else None
            result_dict["point"] = click_point  # can be none

        # set result status
        if result_dict["bbox"] or result_dict["point"]:
            result_status = "positive"
        elif "Target does not exist".lower() in response.lower():
            result_status = "negative"
        else:
            result_status = "wrong_format"
        result_dict["result"] = result_status

        return result_dict


class OSAtlas7BVLLMModel():
    def load_model(self, model_name_or_path="OS-Copilot/OS-Atlas-Base-7B", server_url="http://localhost:8000/v1"):
        self.model_name_or_path = model_name_or_path
        self.client = openai.Client(
            base_url=server_url,
            api_key="token-abc123"  # arbitrary for vllm servers
        )
        # Setting default generation config
        self.generation_config = {
            "max_length": 2048,
            "do_sample": False,
            "temperature": 0.0
        }

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)


    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."

        prompt_origin = 'In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with bbox)?"'
        full_prompt = prompt_origin.format(instruction)

        chat_response = self.client.chat.completions.create(
            model=self.model_name_or_path,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + convert_pil_image_to_base64(image)
                            }
                        },
                        {"type": "text", "text": full_prompt},
                    ],
                }
            ],
            temperature=self.generation_config["temperature"],
            top_p=0.8,
            max_tokens=self.generation_config["max_length"],
            extra_body={
                "skip_special_tokens": False
            }
        )
        response = chat_response.choices[0].message.content
        print(response)


        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        if '<|box_start|>' in response and '<|box_end|>' in response:
            pred_bbox = extract_bbox(response)
            if pred_bbox is not None:
                (x1, y1), (x2, y2) = pred_bbox
                pred_bbox = [pos / 1000 for pos in [x1, y1, x2, y2]]
                x1, y1, x2, y2 = pred_bbox
                if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
                    click_point = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
                    
                    result_dict["bbox"] = pred_bbox
                    result_dict["point"] = click_point
        else:
            print('---------------')
            print(response)
            click_point = pred_2_point(response)
            if click_point is not None:
                x, y = click_point
                if 0 <= x <= 1 and 0 <= y <= 1:
                    click_point = [x / 1000, y / 1000]
                    result_dict["point"] = click_point  # can be none
        
        return result_dict