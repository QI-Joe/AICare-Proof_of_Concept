from transformers import (
    Wav2Vec2ForCTC, Wav2Vec2Processor, 
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoModelForCausalLM, AutoTokenizer)
import torch
import os
from typing import Optional, Any, Tuple, List, Dict
import numpy as np
import re
from utils.log import get_logger

openai_whisper_small = "ASR_model/openai_whisper_small"
openai_whisper_tiny = "openai/whisper-tiny"
model_general_path = r"./ASR_model/"

qwen_llm_tiny = "KingNish/Qwen2.5-0.5b-Test-ft"
logger = get_logger()

def store_modelin_local(model: Optional[WhisperForConditionalGeneration | Wav2Vec2ForCTC], 
    processor: Optional[WhisperProcessor | Wav2Vec2Processor], model_name: str):
    general_path = r"./ASR_model/"
    speicfic_path = os.path.join(general_path, model_name)
    os.makedirs(speicfic_path, exist_ok=True)

    processor.save_pretrained(speicfic_path)
    model.save_pretrained(speicfic_path)
    return True

def is_call_openai_whisper(text):
    # 忽略大小写，匹配 whisper 和 small 两个词，顺序不限，允许连接符 _, - 或无连接
    pattern = re.compile(r'whisper[-_]?small|small[-_]?whisper', re.I)
    return bool(pattern.search(text))

def load_model_test(model_name: str) -> Tuple[Optional[Wav2Vec2ForCTC | WhisperForConditionalGeneration], 
    Optional[Wav2Vec2Processor | WhisperProcessor], torch.device]:
    global model_general_path
    specific_path = os.path.join(model_general_path, model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(specific_path):
        logger.info("Model path does not exist. Auto Download openai whisper small.")
        huggingface_model, auto_model_name = "openai/whisper-small", "openai-whisper-small"
        processor = WhisperProcessor.from_pretrained(huggingface_model)
        model = WhisperForConditionalGeneration.from_pretrained(huggingface_model)
        store_modelin_local(model, processor, auto_model_name)
        logger.info(f"Model downloaded and stored locally at ASR_model/{auto_model_name}.")
        print(f"Model downloaded and stored locally at ASR_model/{auto_model_name}.")

        model = model.to(device)
        return model, processor, device
    
    if is_call_openai_whisper(model_name):
        processor = WhisperProcessor.from_pretrained(specific_path)
        model = WhisperForConditionalGeneration.from_pretrained(specific_path)
    else:
        raise ValueError("Model not recognized.")
    
    # Move model to GPU if available (processor stays on CPU - it's just a tokenizer)
    model = model.to(device)
        
    return model, processor, device

def check_model_status(model: torch.nn.Module, model_name: str) -> Dict[str, Any]:
    """Check and return model status information"""
    device = next(model.parameters()).device
    is_cuda = next(model.parameters()).is_cuda
    dtype = next(model.parameters()).dtype
    
    status = {
        "model_name": model_name,
        "device": str(device),
        "is_cuda": is_cuda,
        "dtype": str(dtype),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    
    if torch.cuda.is_available():
        status["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(device) / 1024**2
        status["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(device) / 1024**2
        status["gpu_name"] = torch.cuda.get_device_name(device)
    
    return status

class ASRModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model, self.processor, self.device = load_model_test(self.model_name)
        
    def get_model(self) -> Tuple[Any, Any]:
        return self.model, self.processor
    
    def check_model_status(self):
        return check_model_status(self.model, self.model_name)

    def process(self, audio: List[float], sp_rate) -> str:
        """For input audio data processing

        Args:
            audio (List[float]): in future maybe a dictionary.
        """
        if is_call_openai_whisper(self.model_name):
            return self.openai_whisper_process(audio, sp_rate)
        raise KeyError("Input model name got wrong.")

    def openai_whisper_process(self, audio: np.ndarray, sp_rate: float) -> str:
        with torch.no_grad():
            encoder_feature = self.processor(audio, sampling_rate=sp_rate, return_tensors="pt").input_features
            encoder_feature = encoder_feature.to(self.device)
            predicts_ids = self.model.generate(encoder_feature)
        
        transcriptions = self.processor.batch_decode(predicts_ids, skip_special_tokens=True)
        return transcriptions[0] if transcriptions else ""

def load_llm_model_safe(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    global model_general_path, qwen_llm_tiny
    
    speicfic_path = os.path.join(model_general_path, model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("The loading path is", speicfic_path)
    
    if not os.path.exists(speicfic_path):
        auto_model_name, auto_model_path = "qwen0.5b", qwen_llm_tiny
        model = AutoModelForCausalLM.from_pretrained(auto_model_path)
        processor = AutoTokenizer.from_pretrained(auto_model_path)
        
        logger.info(f"For LLM, you required LLM is not found, switch to using qwen2.5 variant {auto_model_name}")
        print(f"For LLM, you required LLM is not found, switch to using qwen2.5 variant {auto_model_name}")
        
        store_modelin_local(model, processor, auto_model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(speicfic_path)
        processor = AutoTokenizer.from_pretrained(speicfic_path)
        
    model.to(device)
    return model, processor, device
    
    
class LLModel:
    def __init__(self, model_name: str):
        self.mname = model_name
        self.model, self.processor, self.device = load_llm_model_safe(self.mname)
    
    def get_model(self) -> Tuple[AutoModelForCausalLM | Any, AutoTokenizer | Any]:
        return self.model, self.processor
    
    def check_module_status(self):
        return check_model_status(self.model, self.mname)
    
    def process(self, prompt: str):
        message = [
            {"role": "system", "content": "You are a helpful assistant for general knowledge and support long-term chatting. You can play multiple role in positive as required by user, with professional knowledge to answer. While provide emotion pathatic chatting support."},
            {"role": "user", "content": prompt}
        ]
        text = self.processor.apply_chat_template(
            message,
            tokenize=False, 
            add_generation_prompt=True
        )
        # need to run on GPU-based device, Mac infernce too slow
        model_inputs = self.processor([text], return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(
            **model_inputs,
            max_new_tokens = 512,
        )
        logger.info("[LLM process] Step out from LLM processing order...")
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generate_ids)
        ]
        logger.info("[LLM process] Step in decode procedure...")
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.info("[LLM process] response would be...")
        print("The response is", response)
        
        return response
    
