import torch
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import json
import base64
import re
import os
from io import BytesIO
from PIL import Image

from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def get_qwen2_5vl_prompt_msg(image, instruction, screen_width, screen_height):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant."
                },
                {
                    "type": "text",
                    "text": """


# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "computer_use", "name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\n* The screen's resolution is {{screen_width}}x{{screen_height}}.\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\n* `type`: Type a string of text on the keyboard.\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button.\n* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n* `right_click`: Click the right mouse button.\n* `middle_click`: Click the middle mouse button.\n* `double_click`: Double-click the left mouse button.\n* `scroll`: Performs a scroll of the mouse scroll wheel.\n* `wait`: Wait specified seconds for the change to happen.\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], "type": "string"}, "keys": {"description": "Required only by `action=key`.", "type": "array"}, "text": {"description": "Required only by `action=type`.", "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.", "type": "array"}, "pixels": {"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"}, "time": {"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>""".replace("{{screen_width}}", str(screen_width)).replace("{{screen_height}}", str(screen_height))
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    # "min_pixels": 3136,
                    # "max_pixels": 12845056,
                    "image_url": {
                        "url": "data:image/png;base64," + convert_pil_image_to_base64(image)
                    }
                },
                {
                    "type": "text",
                    "text": instruction
                }
            ]
        }
    ]



GUIDED_PROMPT = """<|im_start|>system
You are a helpful assistant.


# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "computer_use", "name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is {{screen_width}}x{{screen_height}}.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button.
* `middle_click`: Click the middle mouse button.
* `double_click`: Double-click the left mouse button.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], "type": "string"}, "keys": {"description": "Required only by `action=key`.", "type": "array"}, "text": {"description": "Required only by `action=type`.", "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.", "type": "array"}, "pixels": {"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"}, "time": {"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{{instruction}}<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": ["""


class Qwen2_5VLModel():
    def load_model(self, model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        # Setting default generation config
        self.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_length=2048,
            do_sample=False,
            temperature=0.0
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)

    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."

        # Calculate the real image size sent into the model
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
            min_pixels=self.processor.image_processor.min_pixels,
            # max_pixels=self.processor.image_processor.max_pixels,
            max_pixels=99999999,
        )
        print("Resized image size: {}x{}".format(resized_width, resized_height))
        resized_image = image.resize((resized_width, resized_height))

        messages = get_qwen2_5vl_prompt_msg(image, instruction, resized_width, resized_height)

        # Preparation for inference
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        guide_text = "<tool_call>\n{\"name\": \"computer_use\", \"arguments\": {\"action\": \"left_click\", \"coordinate\": ["
        # guide_text = "<tool_call>\n{\"name\": \"computer_use\", \"arguments\": {\"action\": \"mouse_move\", \"coordinate\": ["
        text_input = text_input + guide_text
        
        inputs = self.processor(
            text=[text_input],
            images=[resized_image],
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        print("Len: ", len(inputs.input_ids[0]))
        generated_ids = self.model.generate(**inputs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        response = guide_text + response
        cut_index = response.rfind('}')
        if cut_index != -1:
            response = response[:cut_index + 1]
        print(response)


        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        # Parse action and visualize
        try:
            action = json.loads(response.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
            coordinates = action['arguments']['coordinate']
            if len(coordinates) == 2:
                point_x, point_y = coordinates
            elif len(coordinates) == 4:
                x1, y1, x2, y2 = coordinates
                point_x = (x1 + x2) / 2
                point_y = (y1 + y2) / 2
            else:
                raise ValueError("Wrong output format")
            print(point_x, point_y)
            result_dict["point"] = [point_x / resized_width, point_y / resized_height]  # Normalize predicted coordinates
        except (IndexError, KeyError, TypeError, ValueError) as e:
            pass
        
        return result_dict


    def ground_allow_negative(self, instruction, image):
        raise NotImplementedError()


class CustomQwen2_5VL_VLLM_Model():
    def __init__(self):
        # Check if the current process is daemonic.
        from multiprocessing import current_process
        process = current_process()
        if process.daemon:
            print("Latest vllm versions spawns children processes, therefore can not be started in a daemon process. Are you using multiprocess.Pool? Try multiprocess.Process instead.")

    def load_model(self, model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct", max_pixels=99999999):  #2007040
        from vllm import LLM
        self.max_pixels = max_pixels
        self.model = LLM(
            model_name_or_path,
            gpu_memory_utilization=0.99,
            max_num_seqs=16,
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": self.max_pixels,
            },
        )

    def set_generation_config(self, **kwargs):
        pass

    def ground_only_positive(self, instruction, image):
        from vllm import SamplingParams
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."

        # Calculate the real image size sent into the model
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=14 * 2,
            min_pixels=28 * 28,
            # max_pixels=self.processor.image_processor.max_pixels,
            max_pixels=self.max_pixels,
        )
        print("Resized image size: {}x{}".format(resized_width, resized_height))
        resized_image = image.resize((resized_width, resized_height))

        inputs = {
            "prompt": GUIDED_PROMPT.replace("{{screen_width}}", str(resized_width)).replace("{{screen_height}}", str(resized_height)).replace("{{instruction}}", instruction),
            "multi_modal_data": {"image": resized_image}
        }

        generated = self.model.generate(inputs, sampling_params=SamplingParams(temperature=0.0, max_tokens=100))

        response = generated[0].outputs[0].text.strip()
        print(response)
        response = """<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [""" + response

        cut_index = response.rfind('}')
        if cut_index != -1:
            response = response[:cut_index + 1]
        print(response)


        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        # Parse action and visualize
        try:
            action = json.loads(response.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
            coordinates = action['arguments']['coordinate']
            if len(coordinates) == 2:
                point_x, point_y = coordinates
            elif len(coordinates) == 4:
                x1, y1, x2, y2 = coordinates
                point_x = (x1 + x2) / 2
                point_y = (y1 + y2) / 2
            else:
                raise ValueError("Wrong output format")
            print(point_x, point_y)
            result_dict["point"] = [point_x / resized_width, point_y / resized_height]  # Normalize predicted coordinates
        except (IndexError, KeyError, TypeError, ValueError) as e:
            pass
        
        return result_dict


    def ground_allow_negative(self, instruction, image):
        raise NotImplementedError()
