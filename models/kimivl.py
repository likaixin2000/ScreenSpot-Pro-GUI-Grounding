import os
import re

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, GenerationConfig
from vllm import LLM, SamplingParams



class KimiVL_VLLM_Model():
    def load_model(self, model_name_or_path="moonshotai/Kimi-VL-A3B-Thinking-2506", device="cuda"):
        self.model = LLM(
            model_name_or_path,
            trust_remote_code = True,
            max_num_seqs=8,
            max_model_len=131072,
            limit_mm_per_prompt={"image":256}
        )
        self.processor = AutoProcessor.from_pretrained(model_name_or_path,trust_remote_code=True)
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
        # Generate response
        outputs = self.model.generate([{"prompt": inputs, "multi_modal_data": {"image": image}}], sampling_params=SamplingParams(**self.override_generation_config))
        response = outputs[0].outputs[0].text
        # print("--------")
        # print(response)
        # print("--------")

        def extract_thinking_and_summary(text: str, bot: str = "◁think▷", eot: str = "◁/think▷") -> str:
            if bot in text and eot not in text:
                return ""
            if eot in text:
                return text[text.index(bot) + len(bot):text.index(eot)].strip(), text[text.index(eot) + len(eot) :].strip()
            return "", text

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
