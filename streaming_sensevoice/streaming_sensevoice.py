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

from typing import List

import torch
from asr_decoder import CTCDecoder
from funasr.frontends.wav_frontend import load_cmvn
from online_fbank import OnlineFbank
import numpy as np

from .sensevoice import SenseVoiceSmall


sensevoice_models = {}


class StreamingSenseVoice:
    def __init__(
        self,
        chunk_size: int = 4,
        padding: int = 8,
        beam_size: int = 3,
        contexts: List[str] = None,
        language: str = "zh",
        textnorm: bool = False,
        device: str = "cpu",
        model: str = "iic/SenseVoiceSmall",
        max_history: int = 0,
    ):
        """
        Args:
        language:
            If not empty, then valid values are: auto, zh, en, ja, ko, yue
        textnorm:
            True to enable inverse text normalization; False to disable it.
        max_history:
            Max number of feature frames to retain for encoder context.
            0 (default) means unlimited — the encoder sees all past frames
            with full bidirectional attention, matching the training regime.
            Set to a positive value to bound memory/computation for long audio.
        """
        self.device = device
        self.model, kwargs = self.load_model(model=model, device=device)
        # language query
        language = self.model.lid_dict[language]
        language = torch.LongTensor([[language]]).to(self.device)
        language = self.model.embed(language).repeat(1, 1, 1)
        # text normalization query
        textnorm = self.model.textnorm_dict["withitn" if textnorm else "woitn"]
        textnorm = torch.LongTensor([[textnorm]]).to(self.device)
        textnorm = self.model.embed(textnorm).repeat(1, 1, 1)
        # event and emotion query
        event_emo = self.model.embed(torch.LongTensor([[1, 2]]).to(self.device)).repeat(
            1, 1, 1
        )
        self.query = torch.cat((language, event_emo, textnorm), dim=1)
        # features
        self.input_size = kwargs["input_size"]
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
            self.decoder = CTCDecoder()

        self.chunk_size = chunk_size
        self.max_history = max_history
        self.zeros = np.zeros((1, self.input_size), dtype=float)
        self.feature_buffer = []
        self._last_decoded_frames = 0
        # Rich label reverse mappings (SenseVoice outputs emotion/language/event
        # in the first 4 CTC positions)
        self.lang_map = {v: k for k, v in self.model.lid_int_dict.items()}
        # lid_int_dict maps token IDs (24884, etc.) to internal IDs (3, 4, etc.)
        # lid_dict maps names to internal IDs. Build a reverse from token ID to name.
        _id_to_name = {v: k for k, v in self.model.lid_dict.items()}
        _id_to_name[0] = "auto"
        self.lang_map = {
            token_id: _id_to_name.get(internal_id, "unknown")
            for token_id, internal_id in self.model.lid_int_dict.items()
        }
        self.emo_map = {v: k for k, v in self.model.emo_dict.items()}

    @staticmethod
    def load_model(model: str, device: str) -> tuple:
        key = f"{model}-{device}"
        if key not in sensevoice_models:
            model, kwargs = SenseVoiceSmall.from_pretrained(model=model, device=device)
            model = model.to(device)
            model.eval()
            sensevoice_models[key] = (model, kwargs)
        return sensevoice_models[key]

    def reset(self):
        self.decoder.reset()
        self.fbank = OnlineFbank(window_type="hamming")
        self.feature_buffer = []
        self._last_decoded_frames = 0

    def decode(self, times, tokens):
        times_ms = []
        for step, token in zip(times, tokens):
            if len(self.tokenizer.decode(token).strip()) == 0:
                continue
            times_ms.append(step * 60)
        return times_ms, self.tokenizer.decode(tokens)

    def _run_encoder(self, frames_tensor):
        speech = frames_tensor.unsqueeze(0).to(self.device)
        speech = torch.cat((self.query, speech), dim=1)
        speech_lengths = torch.tensor([speech.shape[1]], device=self.device)
        encoder_out, _ = self.model.encoder(speech, speech_lengths)
        log_probs = self.model.ctc.log_softmax(encoder_out)[0]
        # First 4 positions carry rich labels: language, emotion, event, textnorm
        rich_tokens = log_probs[:4].argmax(dim=-1).tolist()
        return log_probs[4:], rich_tokens

    def streaming_inference(self, audio, is_last):
        self.fbank.accept_waveform(audio, is_last)
        features = self.fbank.get_lfr_frames(
            neg_mean=self.neg_mean, inv_stddev=self.inv_stddev
        )
        if is_last and len(features) == 0:
            features = self.zeros

        self.feature_buffer.extend(
            torch.unbind(torch.tensor(features, dtype=torch.float32), dim=0)
        )

        # Trim history to bound computation
        if self.max_history > 0 and len(self.feature_buffer) > self.max_history:
            trim = len(self.feature_buffer) - self.max_history
            self.feature_buffer = self.feature_buffer[-self.max_history :]
            self._last_decoded_frames = max(0, self._last_decoded_frames - trim)

        # Only yield when enough new frames have arrived (or is_last)
        new_frames = len(self.feature_buffer) - self._last_decoded_frames
        if new_frames < self.chunk_size and not is_last:
            return

        self._last_decoded_frames = len(self.feature_buffer)

        # Run encoder on all accumulated features with full bidirectional attention
        frames_tensor = torch.stack(self.feature_buffer)
        probs, rich_tokens = self._run_encoder(frames_tensor)

        # Decode from scratch — the encoder has full context, so re-decoding
        # gives the best result. The CTC decoder is lightweight.
        self.decoder.reset()
        if self.beam_size > 1:
            res = self.decoder.ctc_prefix_beam_search(
                probs, beam_size=self.beam_size, is_last=is_last
            )
            times_ms, text = self.decode(res["times"][0], res["tokens"][0])
        else:
            res = self.decoder.ctc_greedy_search(probs, is_last=is_last)
            times_ms, text = self.decode(res["times"], res["tokens"])

        rich = {}
        lang_id = rich_tokens[0]
        if lang_id in self.lang_map:
            rich["language"] = self.lang_map[lang_id]
        emo_id = rich_tokens[1]
        if emo_id in self.emo_map:
            rich["emotion"] = self.emo_map[emo_id]
        yield {"timestamps": times_ms, "text": text, "rich": rich}