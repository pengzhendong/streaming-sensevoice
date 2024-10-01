import soundfile as sf
import torch

from asr_decoder import CTCDecoder
from online_fbank import OnlineFbank

from sensevoice import from_pretrained


model = from_pretrained()
samples, sr = sf.read("test_16k.wav")
samples = (samples * 32768).tolist()
fbank = OnlineFbank(window_type="hamming")
decoder = CTCDecoder("contexts.txt", model.symbol_table, model.bpemodel)

chunk_size = 10
padding = 8
idx = 0
step = int(0.1 * sr)
chunk_feats = torch.zeros((chunk_size + 2 * padding, 560))
for i in range(0, len(samples), step):
    is_last = i + step >= len(samples)
    fbank.accept_waveform(samples[i : i + step], is_last)
    feats = fbank.get_lfr_frames(neg_mean=model.neg_mean, inv_stddev=model.inv_stddev)
    if feats is None:
        continue
    for feat in torch.unbind(torch.tensor(feats), dim=0):
        idx += 1
        chunk_feats = torch.roll(chunk_feats, -1, dims=0)
        chunk_feats[-1, :] = feat
        if idx < chunk_size + padding:
            continue
        if (idx - padding) % chunk_size != 0 and not is_last:
            continue
        x = model.inference(chunk_feats)[padding : padding + chunk_size]
        res = decoder.ctc_prefix_beam_search(x, beam_size=3, is_last=is_last)
        print("timestamps:", [i * 60 / 1000 for i in res["times"][0]])
        print("text:", model.tokenizer.decode(res["tokens"][0]))
