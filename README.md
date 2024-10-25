# streaming-sensevoice

Streaming SenseVoice processes inference in chunks of [SenseVoice](https://github.com/FunAudioLLM/SenseVoice).

## Usage

- transcribe wav file

```bash
$ python main.py
```

![](images/screenshot.png)

- transcribe from microphone

```bash
$ python realtime.py
```

- transcribe from websocket

A basic WebSocket service built with `Record.JS` and `FastAPI`; the frontend uses `MP3` format to transmit audio information to reduce latency and increase stability.

```bash
pip install -r requirements-ws-demo.txt
python realtime_ws_server_demo.py

# check cli options
python realtime_ws_server_demo.py --help
```
