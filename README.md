# EduVision-AI

EduVision-AI is a smart classroom assistant project with separate computer vision, speech recognition, and NLP modules.

The current integrated workflow lets you ask a question through your laptop microphone, transcribe it with Whisper, detect whether the transcript contains a question, and classify the question into an academic subject.

## Modules

- Computer Vision: [src/cv_module/README.md](src/cv_module/README.md)
- Speech: [src/speech_module/README.md](src/speech_module/README.md)
- NLP: [src/nlp_module/README.md](src/nlp_module/README.md)

## Setup

From the project root:

```bash
python -m pip install -r requirements.txt
```

Whisper also needs `ffmpeg` installed on your system.

If `transformers.pipeline` fails with a `torchvision::nms` error, reinstall matching PyTorch packages in the same environment:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

## Optional GPU Setup

For NVIDIA GPUs, install the CUDA packages after the normal requirements:

```bash
python -m pip install -r requirements-gpu.txt
```

Verify that both PyTorch and ONNX Runtime can see the GPU:

```bash
python -c "import torch, onnxruntime as ort; ort.preload_dlls(); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(ort.get_available_providers())"
```

## Run Push-To-Talk Subject Classification

Use the root CLI for the full microphone workflow:

```bash
python main.py
```

Flow:

1. Press Enter to start recording.
2. Ask your question.
3. Press Enter again to stop recording.
4. The app prints the transcript, detected question sentences, and predicted subject.

Example result:

```text
Transcript:
What is the derivative of x squared?

Detected sentences:
  [1] QUESTION: What is the derivative of x squared?

Subject classification:
  What is the derivative of x squared?
    Subject: Math (score: 0.93)
```

## Run The FastAPI Services

Run the speech and NLP APIs in separate terminals.

Speech API:

```bash
python -m uvicorn speech_module.api:app --app-dir src --host 127.0.0.1 --port 8000
```

NLP API:

```bash
python -m uvicorn nlp_module.classifier:app --app-dir src --host 127.0.0.1 --port 8001
```

Health checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8001/health
```

Classify text with the NLP API:

```bash
curl -X POST http://127.0.0.1:8001/stt-result \
  -H "Content-Type: application/json" \
  -d '{"text":"What is the derivative of x squared?","language":"en"}'
```

Transcribe an audio file with the speech API:

```bash
curl -X POST http://127.0.0.1:8000/transcribe-file \
  -F "file=@question.wav"
```

## Tests

Run all tests:

```bash
python -m pytest -q
```

Run focused module tests:

```bash
python -m pytest tests/test_nlp_module_api.py -q
python -m pytest tests/test_speech_module_api.py tests/test_speech_module_text_processing.py -q
python -m pytest tests/test_main_integration.py -q
```
