import numpy as np
from scipy.io import wavfile
import soundfile as sf

def parse_s3_uri(uri):
    bucket, key = uri[len('s3://'):].split('/', 1)
    return bucket, key

def write_wav(path, data, sample_rate):
    sf.write(path, data, sample_rate, subtype="PCM_24")

def load_wav(path):
    data, samplerate = sf.read(path, always_2d=True)
    # Reamp playback should be mono (dual-mono stereo files carry no extra info here).
    mono = data[:, 0].astype(np.float32)
    return mono, samplerate

# def load_wav(path):
#     """Load a wav file, return (sample_rate, mono float64 array in [-1, 1])."""
#     sr, raw = wavfile.read(path)
#     data = raw.astype(np.float64)
#     if np.issubdtype(raw.dtype, np.integer):
#         data = data / float(2 ** (raw.dtype.itemsize * 8 - 1))
#     if data.ndim > 1:
#         data = data.mean(axis=1)
#     return sr, data