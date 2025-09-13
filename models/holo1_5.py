import base64
import os
from io import BytesIO
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


class TextContentChunk(BaseModel):
    text: str
    type: Literal["text"] = "text"


class ImageContentChunk(BaseModel):
    base64: str
    type: Literal["image"] = "image"


class ImageUrl(BaseModel):
    url: str


class ImageUrlContentChunk(BaseModel):
    image_url: ImageUrl
    type: Literal["image_url"] = "image_url"


Chunks = TextContentChunk | ImageContentChunk | ImageUrlContentChunk


class UserMessage(BaseModel):
    content: list[Chunks]
    role: Literal["user"] = "user"


class AssistantMessage(BaseModel):
    content: str
    role: Literal["assistant"] = "assistant"


ChatMessage = UserMessage | AssistantMessage


class ChatExample(BaseModel):
    messages: list[ChatMessage]
    response_format: Any | None = None


class ClickAbsoluteAction(BaseModel):
    """Click at absolute coordinates."""

    action: Literal["click_absolute"] = "click_absolute"
    x: int = Field(description="The x coordinate, number of pixels from the left edge.")
    y: int = Field(description="The y coordinate, number of pixels from the top edge.")


class Holo1_5Parameters(BaseModel):
    # Model parameters to mimick what will be on Huggingface after upload

    # Reize params
    max_pixels: int = 2560 * 1440
    min_pixels: int = 4 * 28 * 28
    factor: int = 14 * 2


class VLLMModel:
    def __init__(self):
        self.params = Holo1_5Parameters()
        pass

    def load_model(self, model_name_or_path):
        self.max_pixels = self.params.max_pixels

        self.model = LLM(
            model_name_or_path,
            gpu_memory_utilization=0.95,
            max_num_seqs=16,
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": self.max_pixels,
            },
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            dtype="bfloat16",
            max_model_len=32768
        )


class Holo1_5Model(VLLMModel):
    def __init__(self):
        super().__init__()
        self.prompt = """Localize an element on the GUI image according to the provided target and output a click position.
          * Only output the click position, do not output any other text.
          * The click position should be in the format 'Click(x, y)' with x: num pixels from the left edge and y: num pixels from the top edge
          Your target is:"""

    def load_model(self, model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct"):
        super().load_model(model_name_or_path)

    def set_generation_config(self, **kwargs):
        pass

    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert("RGB")
        assert isinstance(image, Image.Image), "Invalid input image."

        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=self.params.factor,
            min_pixels=self.params.min_pixels,
            max_pixels=self.params.max_pixels,
        )
        image = image.resize((resized_width, resized_height))

        messages = [
            UserMessage(
                content=[
                    ImageUrlContentChunk(
                        image_url=ImageUrl(url="data:image/png;base64," + convert_pil_image_to_base64(image))
                    ),
                    TextContentChunk(text=self.prompt + "\n" + instruction),
                ]
            ),
        ]
        message_dict = [message.model_dump() for message in messages]

        json_schema = ClickAbsoluteAction.model_json_schema()
        guided_decoding_params = GuidedDecodingParams(json=json_schema)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=32, guided_decoding=guided_decoding_params)

        generated = self.model.chat(message_dict, sampling_params=sampling_params)

        response = generated[0].outputs[0].text.strip()

        try:  # Sometimes response is "{". When this happens, we retry with a higher temperature.
            click_action = ClickAbsoluteAction.model_validate_json(response)
        except ValidationError as e:
            backup_sampling_params = SamplingParams(
                temperature=1.0, max_tokens=32, guided_decoding=guided_decoding_params
            )
            for _ in range(10):
                generated = self.model.chat(message_dict, sampling_params=backup_sampling_params)
                response = generated[0].outputs[0].text.strip()
                try:
                    click_action = ClickAbsoluteAction.model_validate_json(response)
                    break
                except ValidationError as e:
                    return {
                        "result": "positive",
                        "format": "x1y1x2y2",
                        "raw_response": response,
                        "bbox": None,
                        "point": None,
                    }

        relative_x = click_action.x / resized_width
        relative_y = click_action.y / resized_height

        return {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": [relative_x, relative_y],
        }
