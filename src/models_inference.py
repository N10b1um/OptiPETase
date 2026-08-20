import subprocess
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class ESM2EmbeddingExtractor:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.embedding_dim = self.model.config.hidden_size
        self.aa_list = [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        ]
        self.aa_tokens = [
            self.tokenizer.convert_tokens_to_ids(aa) for aa in self.aa_list
        ]

    def extract_features(self, wt_seq: str) -> dict[str, Any]:
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
        seq_log_probs = log_probs[0, 1:-1, self.aa_tokens].cpu().numpy()

        zero_shot_scores: dict[int, dict[str, float]] = {}
        for i, wt_aa in enumerate(wt_seq):
            if wt_aa not in self.aa_list:
                continue
            wt_idx = self.aa_list.index(wt_aa)
            wt_log_prob = seq_log_probs[i, wt_idx]

            pos_scores: dict[str, float] = {}
            for j, mut_aa in enumerate(self.aa_list):
                mut_log_prob = seq_log_probs[i, j]
                pos_scores[mut_aa] = float(mut_log_prob - wt_log_prob)
            zero_shot_scores[i] = pos_scores

        return {
            "embeddings": embeddings,
            "log_likelihoods": seq_log_probs,
            "zero_shot_scores": zero_shot_scores,
        }

    def compute_mutation_fitness(self, wt_seq: str, pos_idx: int, mut_aa: str) -> float:
        features = self.extract_features(wt_seq)
        scores = features["zero_shot_scores"].get(pos_idx, {})
        return float(scores.get(mut_aa, 0.0))


class LigandMPNNRunner:
    def __init__(
        self,
        model_type: str = "ligand_mpnn",
        number_of_batches: int = 1,
        batch_size: int = 16,
        temperature: float = 0.1,
        device: str = "cpu",
        catalytic_triad_pdb_ids: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = {
            "model_type": model_type,
            "number_of_batches": number_of_batches,
            "batch_size": batch_size,
            "temperature": temperature,
            "device": device,
            **kwargs,
        }
        if catalytic_triad_pdb_ids is None:
            self.catalytic_triad_pdb_ids = [160, 206, 237]
        else:
            self.catalytic_triad_pdb_ids = catalytic_triad_pdb_ids

    def generate_candidates(
        self,
        input_pdb: str | Path,
        wt_seq: str,
        pdb_to_idx: dict[int, int],
        out_dir: str | Path = "results/",
    ) -> list[dict[str, Any]]:
        input_path = Path(input_pdb)
        if not input_path.exists():
            raise FileNotFoundError(f"Input .pdb file does not exist: {input_path}")

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        idx_to_pdb = {idx: pdb_res for pdb_res, idx in pdb_to_idx.items()}
        catalytic_indices = {
            pdb_to_idx[r] for r in self.catalytic_triad_pdb_ids if r in pdb_to_idx
        }

        aa_alphabet = [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        ]

        candidates: list[dict[str, Any]] = []

        command = [
            "ligandmpnn",
            "--pdb_path",
            str(input_path),
            "--out_folder",
            str(out_path),
        ]
        for key, value in self.config.items():
            command.append(f"--{key}")
            if isinstance(value, bool):
                command.append("1" if value else "0")
            else:
                command.append(str(value))

        executed = False
        try:
            res = subprocess.run(command, capture_output=True, text=True, check=True)
            if res.returncode == 0:
                executed = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            executed = False

        if not executed:
            for i, wt_aa in enumerate(wt_seq):
                if i in catalytic_indices:
                    continue
                pdb_res = idx_to_pdb.get(i, i + 1)
                for mut_aa in aa_alphabet:
                    if mut_aa != wt_aa:
                        candidates.append(
                            {
                                "pos_idx": i,
                                "pdb_idx": pdb_res,
                                "wt_aa": wt_aa,
                                "mut_aa": mut_aa,
                                "mutation_str": f"{wt_aa}{pdb_res}{mut_aa}",
                            }
                        )
        return candidates
