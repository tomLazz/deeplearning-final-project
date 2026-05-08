from pathlib import Path
import random

import torch
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset


STEMS = ["vocals", "drums", "bass", "guitar", "other"]

STEM_FOLDERS = {
    "vocals": ["Voc"],
    "drums": ["Rhy"],
    "bass": ["Bass"],
    "guitar": ["Gtr"],
    "other": ["Kbs", "Misc", "Orch", "Synth"],
}


class RawStemsDataset(Dataset):
    def __init__(self, root_dir, chunk_size=294400):
        self.root_dir = Path(root_dir)
        self.chunk_size = chunk_size

        self.song_dirs = [p for p in self.root_dir.iterdir() if p.is_dir()]

        if len(self.song_dirs) == 0:
            raise RuntimeError(f"No song folders found in {self.root_dir}")

        print(f"[+] Found {len(self.song_dirs)} songs in {self.root_dir}")

    def __len__(self):
        return len(self.song_dirs)

    def _load_audio(self, path):
        audio, sr = sf.read(path)
        audio = torch.tensor(audio).float()

        if audio.ndim == 1:
            audio = audio.unsqueeze(1).repeat(1, 2)

        audio = audio.T

        return audio

    def _load_stem_group(self, song_dir, stem_name):
        audios = []

        for folder_name in STEM_FOLDERS[stem_name]:
            folder = song_dir / folder_name

            if not folder.exists():
                continue

            files = list(folder.rglob("*.wav")) + list(folder.rglob("*.flac"))

            for file in files:
                audios.append(self._load_audio(file))

        if len(audios) == 0:
            return torch.zeros(2, self.chunk_size)

        min_len = min(audio.shape[-1] for audio in audios)
        audios = [audio[:, :min_len] for audio in audios]

        return torch.stack(audios, dim=0).sum(dim=0)

    def __getitem__(self, idx):
        song_dir = self.song_dirs[idx]

        stem_audio_list = []

        for stem in STEMS:
            audio = self._load_stem_group(song_dir, stem)
            stem_audio_list.append(audio)

        min_len = min(audio.shape[-1] for audio in stem_audio_list)

        if min_len >= self.chunk_size:
            start = random.randint(0, min_len - self.chunk_size)
            end = start + self.chunk_size
            stem_audio_list = [audio[:, start:end] for audio in stem_audio_list]
        else:
            padded = []
            for audio in stem_audio_list:
                pad_len = self.chunk_size - audio.shape[-1]
                audio = F.pad(audio, (0, pad_len))
                padded.append(audio)
            stem_audio_list = padded

        stems = torch.stack(stem_audio_list, dim=0)
        mixture = stems.sum(dim=0)

        return {
            "mixture": mixture,
            "stems": stems
        }