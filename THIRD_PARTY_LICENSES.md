# Third-party licenses

## SingleWordProductionDutch (Neural Interfacing Lab)

Source: <https://github.com/neuralinterfacinglab/SingleWordProductionDutch>, commit `cfb563696a8d44207532e3777ba6c5aabaf68805`.

The following files in `src/` are adapted from that repository. The only edits are compatibility fixes for current
NumPy/SciPy releases (removed aliases), with no change in behaviour; each file's header lists its edits.

| This repository | Upstream file | Edits |
|---|---|---|
| `src/swp_features.py` | `extract_features.py` | `scipy.hanning` → `scipy.signal.windows.hann`; string `scipy.stats.mode` → `np.unique`-based mode with the same tie rule |
| `src/MelFilterBank.py` | `MelFilterBank.py` | `np.float` → `float` |
| `src/reconstructWave.py` | `reconstructWave.py` | `scipy.hanning` → `scipy.signal.windows.hann`; `scipy.zeros`/`scipy.real` → `np.zeros`/`np.real` |

```text
MIT License

Copyright (c) 2021 Neural Interfacing Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```
