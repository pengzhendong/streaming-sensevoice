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

from functools import partial
from typing import List

import torch
from asr_decoder import CTCDecoder
from funasr import AutoModel
from funasr.frontends.wav_frontend import load_cmvn
from online_fbank import OnlineFbank

from .sensevoice import SenseVoiceSmall


sensevoice_models = {}


class StreamingSenseVoice:
    def __init__(
        self,
        chunk_size: int = 10,
        padding: int = 8,
        beam_size: int = 3,
        contexts: List[str] = None,
        device: str = "cpu",
        model: str = "iic/SenseVoiceSmall",
    ):
        self.device = device
        key = f"{model}-{device}"
        if key not in sensevoice_models:
            model, kwargs = SenseVoiceSmall.from_pretrained(model=model)
            model = model.to(device)
            model.eval()
            sensevoice_models[key] = (model, kwargs)
        self.model, kwargs = sensevoice_models[key]
        # features
        cmvn = load_cmvn(kwargs["frontend_conf"]["cmvn_file"]).numpy()
        self.neg_mean, self.inv_stddev = cmvn[0, :], cmvn[1, :]
        self.fbank = OnlineFbank(window_type="hamming")
        # decoder
        self.tokenizer = kwargs["tokenizer"]
        bpe_model = kwargs["tokenizer_conf"]["bpemodel"]
        symbol_table = {}
        for i in range(self.tokenizer.get_vocab_size()):
            symbol_table[self.tokenizer.decode(i)] = i
        if beam_size > 1 and contexts is not None:
            self.beam_size = beam_size
            self.decoder = CTCDecoder(contexts, symbol_table, bpe_model)
        else:
            self.beam_size = 1
            self.decoder = CTCDecoder(symbol_table=symbol_table, bpe_model=bpe_model)

        self.chunk_size = chunk_size
        self.padding = padding
        self.cur_idx = -1
        self.caches_shape = (chunk_size + 2 * padding, kwargs["input_size"])
        self.caches = torch.zeros(self.caches_shape)

    def reset(self):
        self.cur_idx = -1
        self.decoder.reset()
        self.fbank = OnlineFbank(window_type="hamming")
        self.caches = torch.zeros(self.caches_shape)

    def get_size(self):
        effective_size = self.cur_idx + 1 - self.padding
        if effective_size <= 0:
            return 0
        return effective_size % self.chunk_size or self.chunk_size

    def inference(self, speech):
        speech = speech[None, :, :]
        speech_lengths = torch.tensor([speech.shape[1]])
        speech = speech.to(self.device)
        speech_lengths = speech_lengths.to(self.device)

        textnorm_query = self.model.embed(
            torch.LongTensor([[self.model.textnorm_dict["woitn"]]]).to(self.device)
        ).repeat(speech.size(0), 1, 1)
        language_query = self.model.embed(
            torch.LongTensor([[self.model.lid_dict["zh"]]]).to(self.device)
        ).repeat(speech.size(0), 1, 1)
        event_emo_query = self.model.embed(
            torch.LongTensor([[1, 2]]).to(self.device)
        ).repeat(speech.size(0), 1, 1)
        speech = torch.cat(
            (language_query, event_emo_query, textnorm_query, speech), dim=1
        )
        speech_lengths += 4

        encoder_out, _ = self.model.encoder(speech, speech_lengths)
        return self.model.ctc.log_softmax(encoder_out)[0, 4:]

    def streaming_inference(self, audio, is_last):
        self.fbank.accept_waveform(audio, is_last)
        features = self.fbank.get_lfr_frames(
            neg_mean=self.neg_mean, inv_stddev=self.inv_stddev
        )
        if features is None:
            return None
        for feature in torch.unbind(torch.tensor(features), dim=0):
            self.caches = torch.roll(self.caches, -1, dims=0)
            self.caches[-1, :] = feature
            self.cur_idx += 1
            cur_size = self.get_size()
            if cur_size != self.chunk_size and not is_last:
                continue
            probs = self.inference(self.caches)[self.padding :]
            if cur_size != self.chunk_size:
                probs = probs[self.chunk_size - cur_size :]
            if not is_last:
                probs = probs[: self.chunk_size]
            if self.beam_size > 1:
                res = self.decoder.ctc_prefix_beam_search(
                    probs, beam_size=self.beam_size, is_last=is_last
                )
                timestamps = [i * 60 for i in res["times"][0]]
                text = self.tokenizer.decode(res["tokens"][0])
            else:
                res = self.decoder.ctc_greedy_search(probs, is_last=is_last)
                timestamps = [i * 60 for i in res["times"]]
                text = self.tokenizer.decode(res["tokens"])
            yield {"timestamps": timestamps, "text": text}
