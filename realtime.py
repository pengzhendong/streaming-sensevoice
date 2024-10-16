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


def get_size(cur_idx, chunk_size, padding):
    effective_size = cur_idx + 1 - padding
    if effective_size <= 0:
        return 0
    return effective_size % chunk_size or chunk_size


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
                chunk_size = 10
                padding = 8
                chunk_feats = torch.zeros((chunk_size + 2 * padding, 560))
                idx = -1
                decoder.reset()
                fbank = OnlineFbank(window_type="hamming")
            is_last = "end" in speech_dict
            fbank.accept_waveform(speech_samples.tolist(), is_last)
            feats = fbank.get_lfr_frames(
                neg_mean=model.neg_mean, inv_stddev=model.inv_stddev
            )
            if feats is None:
                continue
            for feat in torch.unbind(torch.tensor(feats), dim=0):
                chunk_feats = torch.roll(chunk_feats, -1, dims=0)
                chunk_feats[-1, :] = feat
                idx += 1
                cur_size = get_size(idx, chunk_size, padding)
                if cur_size != chunk_size and not is_last:
                    continue
                x = model.inference(chunk_feats)[padding:]
                if cur_size != chunk_size:
                    x = x[chunk_size - cur_size:]
                if not is_last:
                    x = x[:chunk_size]
                res = decoder.ctc_prefix_beam_search(x, beam_size=3, is_last=is_last)
                print("timestamps:", [i * 60 / 1000 for i in res["times"][0]])
                print("text:", model.tokenizer.decode(res["tokens"][0]))
