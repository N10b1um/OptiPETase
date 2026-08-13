import os
import warnings
from typing import Dict, Any, List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from ligandmpnn import *
from pathlib import Path
import subprocess
import sys

class ESM2EmbeddingExtractor:
    def __init__(self, model_name: str, device: str):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.embedding_dim = self.model.config.hidden_size

    def extract_features(self, wt_seq: str) -> Dict[str, Any]:
        if not wt_seq:
            raise ValueError("The input sequence cannot be empty.")

        inputs = self.tokenizer(wt_seq, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden_state = outputs.hidden_states[-1]
        embeddings = last_hidden_state[0, 1:-1, :].cpu().numpy()

        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        aa_list = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
        aa_tokens = [self.tokenizer.convert_tokens_to_ids(aa) for aa in aa_list]

        seq_log_probs = log_probs[0, 1:-1, aa_tokens].cpu().numpy()

        zero_shot_scores = {}
        for i, wt_aa in enumerate(wt_seq):
            if wt_aa not in aa_list:
                continue
            wt_idx = aa_list.index(wt_aa)
            wt_log_prob = seq_log_probs[i, wt_idx]
            
            pos_scores = {}
            for j, mut_aa in enumerate(aa_list):
                mut_log_prob = seq_log_probs[i, j]
                pos_scores[mut_aa] = float(mut_log_prob - wt_log_prob)
            zero_shot_scores[i] = pos_scores

        return {
            "embeddings": embeddings,
            "log_likelihoods": seq_log_probs,
            "zero_shot_scores": zero_shot_scores
        }


class LigandMPNNRunner:
    def __init__(self,
                model_type: str = "ligand_mpnn", 
                number_of_batches: int = 1,
                batch_size: int = 1,
                temperature: float = 0.1,
                device: str = "cpu",
                **kwargs
    ):
        self.config = {
            "model_type": model_type,
            "number_of_batches": number_of_batches,
            "batch_size": batch_size,
            "temperature": temperature,
            "device": device,
            **kwargs
        }

    def generate_candidates(
            self,
            input_pdb: str | Path,
            out_dir: str | Path = "results/"
    ) -> subprocess.CompletedProcess:
        if not Path(input_pdb).exists():
            raise FileNotFoundError(f"Input .pdb file does not exist: {input_pdb}")
        
        command = [
            "ligandmpnn",
            "--pdb_path", str(input_pdb),
            "--out_folder", str(out_dir),
        ]

        for key, value in self.config.items():
            command.append(f"--{key}")
            if isinstance(value, bool):
                command.append("1" if value else "0")
            else:
                command.append(str(value))

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return result
        
        except subprocess.CalledProcessError as e:
            print(f"Error running ligandmpnn on {input_pdb}:", file = sys.stderr)
            print(e.stderr , file=sys.stderr)
            raise e