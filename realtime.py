# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sounddevice as sd
import torch

from asr_decoder import CTCDecoder
from online_fbank import OnlineFbank
from pysilero import VADIterator

from sensevoice import from_pretrained


device = "mps"
model = from_pretrained(device)
vad_iterator = VADIterator()
decoder = CTCDecoder("contexts.txt", model.symbol_table, model.bpemodel)
devices = sd.query_devices()
if len(devices) == 0:
    print("No microphone devices found")
    sys.exit(0)
print(devices)
default_input_device_idx = sd.default.device[0]
print(f'Use default device: {devices[default_input_device_idx]["name"]}')

samples_per_read = int(0.1 * 16000)
with sd.InputStream(channels=1, dtype="float32", samplerate=16000) as s:
    while True:
        samples, _ = s.read(samples_per_read)
        for speech_dict, speech_samples in vad_iterator(samples[:, 0]):
            if "start" in speech_dict:
                decoder.reset()
                fbank = OnlineFbank(window_type="hamming")
            is_last = "end" in speech_dict
            fbank.accept_waveform(speech_samples.tolist(), is_last)
            if not is_last:
                continue
            feats = fbank.get_lfr_frames(
                neg_mean=model.neg_mean, inv_stddev=model.inv_stddev
            )
            if feats is None:
                continue
            x = model.inference(torch.tensor(feats))
            res = decoder.ctc_prefix_beam_search(x, beam_size=3, is_last=True)
            if len(res["tokens"][0]) > 0:
                print("text:", model.tokenizer.decode(res["tokens"][0]))
