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
import soundfile as sf
from pysilero import VADIterator

from streaming_sensevoice import StreamingSenseVoice


def main():
    model = StreamingSenseVoice()
    vad_iterator = VADIterator(speech_pad_ms=300)

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
                    model.reset()
                is_last = "end" in speech_dict
                for res in model.streaming_inference(speech_samples * 32768, is_last):
                    sf.write("test.wav", vad_iterator.speech_samples, 16000)
                    print(res["timestamps"])
                    print(res["text"])


if __name__ == "__main__":
    main()
