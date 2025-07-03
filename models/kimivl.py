import os
import re
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)  # Use 'spawn' to avoid issues with CUDA in multiprocessing
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, GenerationConfig
from vllm import LLM, SamplingParams


def extract_thinking_and_summary(text: str, bot: str = "◁think▷", eot: str = "◁/think▷") -> str:
    if bot in text and eot not in text:
        return ""
    if eot in text:
        return text[text.index(bot) + len(bot):text.index(eot)].strip(), text[text.index(eot) + len(eot) :].strip()
    return "", text


class KimiVLModel():
    def load_model(self, model_name_or_path="moonshotai/Kimi-VL-A3B-Thinking-2506"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map="cuda", 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

        # Setting default generation config
        self.generation_config = dict()
        self.set_generation_config(
            do_sample=False,
            temperature=0.0,
            max_new_tokens=512,
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)

    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        elif image is None:
            raise ValueError("`image` should be provided.")
        
        messages = [
            {
                "role": "system", 
                "content": "You are a GUI agent. You are given a task and a screenshot of a computer screen. You need to perform a action and pyautogui code to complete the task. Provide your response in this format:\n\n## Action:\nProvide clear, concise, and actionable instructions.\n\n## Code:\nGenerate a corresponding Python code snippet using pyautogui that clicks on the identified UI element using normalized screen coordinates (values between 0 and 1). The script should dynamically adapt to the current screen resolution by converting the normalized coordinates to actual pixel positions."},
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": ""}, 
                    {"type": "text", "text": f"## Task Instruction:\n{instruction}"}
                ]
            }
        ]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

        # Preparation for inference
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_input],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        print("Token length: ", len(inputs.input_ids[0]))
        generated_ids = self.model.generate(**inputs, **self.generation_config)
        response = self.processor.decode(
            generated_ids[0], 
            skip_special_tokens=False, 
            clean_up_tokenization_spaces=False
        ).strip()

        output_format = "--------Thinking--------\n{thinking}\n\n--------Summary--------\n{summary}"

        thinking, summary = extract_thinking_and_summary(response)
        print(output_format.format(thinking=thinking, summary=summary))
        # Extract bounding boxes from the response
        
        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        # Parse action and visualize
        try:
            # Extract bounding boxes from the response
            match = re.search(r"x=(0(?:\.\d+)?|1(?:\.0+)?), y=(0(?:\.\d+)?|1(?:\.0+)?)", summary)
            if match:
                result_dict["point"] = [float(match.group(1)), float(match.group(2))]
                print("Predicted: ", result_dict["point"])  # {'x': 0.204, 'y': 0.149}
            else:
                print("No bounding boxes found in the response.")
        except (IndexError, KeyError, TypeError, ValueError) as e:
            pass
        
        return result_dict


    def ground_allow_negative(self, instruction, image):
        raise NotImplementedError()


class KimiVL_VLLM_Model():
    def load_model(self, model_name_or_path="moonshotai/Kimi-VL-A3B-Thinking-2506", device="cuda"):
        print("Trying to load Kimi-VL model. If you encounter vllm issues, please try setting `VLLM_FLASH_ATTN_VERSION=3` in your environment variables.")
        self.model = LLM(
            model_name_or_path,
            trust_remote_code = True,
            max_num_seqs=1,
            max_model_len=16384,
            tensor_parallel_size=1,
            dtype=torch.bfloat16,
            # limit_mm_per_prompt={"image":1}
        )
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        # Setting default generation config
        self.override_generation_config = dict(max_tokens=8192, temperature=0.0)


    def set_generation_config(self, **kwargs):
        self.override_generation_config.update(kwargs)
        if "max_new_tokens" in self.override_generation_config:
            self.override_generation_config["max_tokens"] = self.override_generation_config["max_new_tokens"]
            del self.override_generation_config["max_new_tokens"]


    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        elif image is None:
            raise ValueError("`image` should be provided.")
        
        messages = [
            {
                "role": "system", 
                "content": "You are a GUI agent. You are given a task and a screenshot of a computer screen. You need to perform a action and pyautogui code to complete the task. Provide your response in this format:\n\n## Action:\nProvide clear, concise, and actionable instructions.\n\n## Code:\nGenerate a corresponding Python code snippet using pyautogui that clicks on the identified UI element using normalized screen coordinates (values between 0 and 1). The script should dynamically adapt to the current screen resolution by converting the normalized coordinates to actual pixel positions."},
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": ""}, 
                    {"type": "text", "text": f"## Task Instruction:\n{instruction}"}
                ]
            }
        ]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        print(inputs)
        # Generate response
        outputs = self.model.generate([{"prompt": inputs, "multi_modal_data": {"image": image}}], sampling_params=SamplingParams(**self.override_generation_config))
        response = outputs[0].outputs[0].text
        # print("--------")
        # print(response)
        # print("--------")

        

        output_format = "--------Thinking--------\n{thinking}\n\n--------Summary--------\n{summary}"

        thinking, summary = extract_thinking_and_summary(response)
        print(output_format.format(thinking=thinking, summary=summary))
    


        bbox = None
        click_point = None
        # Extract bounding boxes from the response
        match = re.search(r"x=(0(?:\.\d+)?|1(?:\.0+)?), y=(0(?:\.\d+)?|1(?:\.0+)?)", summary)
        if match:
            click_point = [float(match.group(1)), float(match.group(2))]
            print(click_point)  # {'x': 0.204, 'y': 0.149}
        else:
            print("No bounding boxes found in the response.")

        result_dict = {
            "result": "positive",
            "bbox": bbox,
            "point": click_point,
            "raw_response": response
        }
        
        return result_dict
