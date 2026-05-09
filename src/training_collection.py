import os
import csv
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

STEMS      = ['vocals', 'drums', 'bass', 'guitar', 'other']
CHUNK_SIZE = 588800   # ~13.3s at 44100Hz


# -----------------------------------------------------------------------
# Collection
# -----------------------------------------------------------------------
def collect_stems(manifest_path, split, rawstems_dir='/content/rawstems',
                  chunk_size=CHUNK_SIZE):
    """
    Returns a dict mapping each stem to a list of (path, start_sample) pairs,
    one entry per non-overlapping chunk per file.
    """
    stem_lists = {s: [] for s in STEMS}

    with open(manifest_path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('split', '').strip().lower() != split:
                continue
            track_id = row['track_id']
            for stem in STEMS:
                if not row.get(stem, '').strip():
                    continue
                path = os.path.join(rawstems_dir, track_id, f'{stem}.flac')
                if not os.path.exists(path):
                    continue
                try:
                    n_frames = sf.info(path).frames
                except Exception:
                    continue  # skip unreadable files
                starts = list(range(0, max(n_frames, chunk_size), chunk_size))
                for start in starts:
                    stem_lists[stem].append((path, start))

    for stem, chunks in stem_lists.items():
        print(f"  {stem:>7s}: {len(chunks)} chunks")

    return stem_lists


def pad_stem_lists(stem_lists):
    """Repeats shorter lists until all are the same length as the longest."""
    max_len = max(len(v) for v in stem_lists.values())
    padded  = {}
    for stem, chunks in stem_lists.items():
        if len(chunks) == 0:
            padded[stem] = []
            continue
        reps = (max_len + len(chunks) - 1) // len(chunks)
        padded[stem] = (chunks * reps)[:max_len]
    return padded, max_len


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------
class RemixDataset(Dataset):
    """
    Each sample independently draws one (path, start) per stem, loads that
    chunk, and sums the stems to produce a synthetic mix.
    """
    def __init__(self, stem_lists, chunk_size=CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.lists, self.length = pad_stem_lists(stem_lists)
        for stem in STEMS:
            np.random.shuffle(self.lists[stem])

    def _load_chunk(self, path, start):
        info     = sf.info(path)
        n_frames = info.frames
        read_len = min(self.chunk_size, max(0, n_frames - start))

        if read_len > 0:
            try:
                audio, _ = sf.read(path, start=start, frames=read_len,
                                    dtype='float32', always_2d=True)
                audio = audio.T                             # (C, T)
                if audio.shape[0] == 1:
                    audio = np.repeat(audio, 2, axis=0)
                audio = audio[:2]
            except Exception:
                audio = np.zeros((2, 0), dtype=np.float32)
        else:
            audio = np.zeros((2, 0), dtype=np.float32)

        # Pad if chunk is shorter than chunk_size (end of file)
        if audio.shape[1] < self.chunk_size:
            audio = np.pad(audio, ((0, 0), (0, self.chunk_size - audio.shape[1])))

        return audio  # (2, chunk_size)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        stems = []
        for stem in STEMS:
            chunks = self.lists[stem]
            if chunks:
                path, start = chunks[idx % len(chunks)]
                audio = self._load_chunk(path, start)
            else:
                audio = np.zeros((2, self.chunk_size), dtype=np.float32)
            stems.append(audio)

        stems_arr = np.stack(stems, axis=0)                # (5, 2, chunk_size)

        if np.random.random() < 0.5:
            stems_arr = stems_arr[:, ::-1, :].copy()       # flip L/R channels

        mix  = stems_arr.sum(axis=0)                       # (2, chunk_size)
        peak = np.abs(mix).max()
        if peak > 1.0:
            stems_arr = stems_arr / peak
            mix       = mix       / peak

        return torch.from_numpy(mix), torch.from_numpy(stems_arr)


# -----------------------------------------------------------------------
# Quick test
# -----------------------------------------------------------------------
if __name__ == '__main__':
    MANIFEST_PATH = '/content/manifest.csv'
    RAWSTEMS_DIR  = '/content/rawstems'

    print('[train]')
    train_lists = collect_stems(MANIFEST_PATH, split='train', rawstems_dir=RAWSTEMS_DIR)
    train_set   = RemixDataset(train_lists)
    print(f'  dataset length: {len(train_set)}')

    mix, stems = train_set[0]
    print(f'  mix shape:   {mix.shape}')
    print(f'  stems shape: {stems.shape}')

    print('[test]')
    test_lists = collect_stems(MANIFEST_PATH, split='test', rawstems_dir=RAWSTEMS_DIR)
    test_set   = RemixDataset(test_lists)
    print(f'  dataset length: {len(test_set)}')
