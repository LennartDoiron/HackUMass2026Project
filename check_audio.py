import wave, sys
import numpy as np

f = wave.open(sys.argv[1])
n, w = f.getnchannels(), f.getsampwidth()
dtype = {2: np.int16, 4: np.int32}[w]
a = np.frombuffer(f.readframes(f.getnframes()), dtype=dtype).reshape(-1, n).astype(float)
full = float(np.iinfo(dtype).max)
for ch in range(n):
    x = a[:, ch]
    print(f"channel {ch}: peak {abs(x).max()/full*100:.2f}%   "
          f"avg {abs(x).mean()/full*100:.3f}%   different values: {len(np.unique(x))}")