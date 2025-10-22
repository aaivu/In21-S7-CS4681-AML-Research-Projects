import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Config, TrainingArguments, Trainer
import numpy as np

class Model:
    """
    Wrapper class for loading and running a WavLM CTC model with a processor.
    Supports inference and text transcription from raw audio arrays.
    """

    def __init__(self, model_dir: str = None, processor_dir: str = None, device: str = None, dropout: int = 0.2):
        """
        Initialize model and processor from pretrained directories or Hugging Face model names.

        Args:
            model_dir (str): Path or HF name of the trained CTC model (e.g., 'microsoft/wavlm-large')
            processor_dir (str): Path or HF name of the processor (tokenizer + feature extractor)
            device (str, optional): 'cuda' or 'cpu'. Auto-detects if not provided.
        """
        self.model = None
        self.processor = None

        if processor_dir:
            self.set_processor(processor_dir)
            if model_dir:
                self.set_model(model_dir, dropout, device)

    def predict_logits(self, batch) -> np.ndarray:
        """
        Run the model forward pass and return logits.
        """
        inputs = self.processor(batch["audio"], sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits
        batch["logits"] = logits.cpu().numpy()
        return batch
    
    def set_model(self, model_dir, dropout, device) -> None:
        if self.model is None:
            self.config = Wav2Vec2Config.from_pretrained(model_dir)
            self.config.ctc_loss_reduction = "mean"
            self.config.pad_token_id = self.processor.tokenizer.pad_token_id
            self.config.final_dropout = dropout
            self.model = Wav2Vec2ForCTC.from_pretrained(
                model_dir,
                config=self.config
            )

            # Auto-select device
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = torch.device(device)
            self.model.to(self.device)
        
    def set_processor(self, processor_dir) -> None:
        if self.processor is None:
            self.processor = Wav2Vec2Processor.from_pretrained(processor_dir)
    
    def get_model(self):
        return self.model
    
    def get_processor(self):
        return self.processor
    
    def prepare(self, batch) -> np.ndarray:
        speech = batch["audio"]["array"]
        batch["input_values"] = self.processor(speech, sampling_rate=16000, return_tensors="np")
        with self.processor.as_target_processor():
            batch["labels"] = self.processor.tokenizer(batch["text"]).input_ids
        return batch
