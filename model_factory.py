def build_model(args):
    model_type = args.model_type
    model_name_or_path = args.model_name_or_path
    if model_type == "kimivl":
        from models.kimivl import KimiVLModel
        model = KimiVLModel()
        if model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    elif model_type == "seeclick":
        from models.seeclick import SeeClickModel
        model = SeeClickModel()
        model.load_model()
    elif model_type == "qwen1vl":
        from models.qwen1vl import Qwen1VLModel
        model = Qwen1VLModel()
        model.load_model()
    elif model_type == "qwen2vl":
        from models.qwen2vl import Qwen2VLModel
        model = Qwen2VLModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    elif model_type == "qwen2_5vl":
        from models.qwen2_5vl import Qwen2_5VLModel
        model = Qwen2_5VLModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
            
    elif model_type == "minicpmv":
        from models.minicpmv import MiniCPMVModel
        model = MiniCPMVModel()
        model.load_model()
    elif model_type == "internvl":
        from models.internvl import InternVLModel
        model = InternVLModel()
        model.load_model()

    elif model_type in ["gpt4o", "gpt4v"]:
        from models.gpt4x import GPT4XModel
        model = GPT4XModel()
        
    elif model_type == "osatlas-4b":
        from models.osatlas4b import OSAtlas4BModel
        model = OSAtlas4BModel()
        model.load_model()
    elif model_type == "osatlas-7b":
        from models.osatlas7b import OSAtlas7BModel
        model = OSAtlas7BModel()
        model.load_model()
    elif model_type == "uground":
        from models.uground import UGroundModel
        model = UGroundModel()
        model.load_model()

    elif model_type == "fuyu":
        from models.fuyu import FuyuModel
        model = FuyuModel()
        model.load_model()
    elif model_type == "showui":
        from models.showui import ShowUIModel
        model = ShowUIModel()
        model.load_model()
    elif model_type == "ariaui":
        from models.ariaui import AriaUIVLLMModel
        model = AriaUIVLLMModel()
        model.load_model()
    elif model_type == "cogagent":
        from models.cogagent import CogAgentModel
        model = CogAgentModel()
        model.load_model()
    elif model_type == "cogagent24":
        from models.cogagent24 import CogAgent24Model
        model = CogAgent24Model()
        model.load_model()

    elif model_type == "screenseeker":
        from models.seeclick_pro import SeeClickProAgent
        from models.osatlas7b import OSAtlas7BVLLMModel
        grounder = OSAtlas7BVLLMModel()
        grounder.load_model()
        model = SeeClickProAgent(grounder=grounder)
    else:
        raise ValueError(f"Unsupported model type {model_type}.")
    model.set_generation_config(temperature=0, max_new_tokens=256)
    return model
    