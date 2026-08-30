# SPDX-License-Identifier: Apache-2.0

# imports for guided decoding tests
import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import librosa
import numpy as np
import openai
import pytest
import soundfile as sf
from openai._base_client import AsyncAPIClient

from vllm.assets.audio import AudioAsset
from vllm.entrypoints.openai import serving_transcription

from ...utils import RemoteOpenAIServer


@pytest.fixture
def mary_had_lamb():
    path = AudioAsset('mary_had_lamb').get_local_path()
    with open(str(path), "rb") as f:
        yield f


@pytest.fixture
def winning_call():
    path = AudioAsset('winning_call').get_local_path()
    with open(str(path), "rb") as f:
        yield f


def _flac_audio(duration_s: float, sample_rate: int = 8000) -> bytes:
    audio = np.zeros(round(duration_s * sample_rate), dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="FLAC")
    return buffer.getvalue()


def _transcription_serving(max_duration_s: float):
    serving = serving_transcription.OpenAIServingTranscription.__new__(
        serving_transcription.OpenAIServingTranscription)
    serving.max_audio_clip_s = max_duration_s
    return serving


@pytest.mark.asyncio
async def test_audio_duration_rejected_before_decode():
    serving = _transcription_serving(max_duration_s=1)
    audio_data = _flac_audio(duration_s=2)
    request = SimpleNamespace(language=None, prompt="")

    assert len(audio_data) < 25 * 1024**2
    with (patch.object(serving_transcription.librosa, "load") as decode,
          pytest.raises(ValueError,
                        match=r"Maximum clip duration \(1s\) exceeded")):
        await serving._preprocess_transcription(request, audio_data)

    decode.assert_not_called()


@pytest.mark.asyncio
async def test_audio_decode_is_bounded_when_metadata_underreports_duration():
    serving = _transcription_serving(max_duration_s=30)
    request = SimpleNamespace(language=None, prompt="")
    audio_file = MagicMock(frames=29, samplerate=1, channels=1)
    audio_file.__enter__.return_value = audio_file

    with (patch.object(serving_transcription.soundfile,
                       "SoundFile",
                       return_value=audio_file),
          patch.object(serving_transcription.librosa,
                       "load",
                       return_value=(MagicMock(), 22050)) as decode,
          patch.object(serving_transcription.librosa,
                       "get_duration",
                       return_value=30.5),
          pytest.raises(ValueError,
                        match=r"Maximum clip duration \(30s\) exceeded")):
        await serving._preprocess_transcription(request, b"audio")

    decode.assert_called_once_with(audio_file, duration=31.0)


@pytest.mark.asyncio
async def test_audio_decoded_size_rejected_before_decode():
    serving = _transcription_serving(max_duration_s=30)
    request = SimpleNamespace(language=None, prompt="")
    frames = serving_transcription.MAX_AUDIO_DECODE_BYTES // 4 + 1
    audio_file = MagicMock(frames=frames,
                           samplerate=frames,
                           channels=1)
    audio_file.__enter__.return_value = audio_file

    with (patch.object(serving_transcription.soundfile,
                       "SoundFile",
                       return_value=audio_file),
          patch.object(serving_transcription.librosa, "load") as decode,
          pytest.raises(ValueError,
                        match="Maximum decoded audio size exceeded")):
        await serving._preprocess_transcription(request, b"audio")

    decode.assert_not_called()


@pytest.mark.asyncio
async def test_audio_at_duration_limit_is_preserved():
    serving = _transcription_serving(max_duration_s=1)
    request = SimpleNamespace(language=None, prompt="keep this prompt")

    prompt, duration = await serving._preprocess_transcription(
        request, _flac_audio(duration_s=1))

    audio, sample_rate = prompt["encoder_prompt"]["multi_modal_data"][
        "audio"]
    assert duration == pytest.approx(1)
    assert librosa.get_duration(y=audio, sr=sample_rate) == pytest.approx(1)
    assert prompt["decoder_prompt"].endswith(request.prompt)


@pytest.mark.asyncio
async def test_basic_audio(mary_had_lamb):
    model_name = "openai/whisper-large-v3-turbo"
    server_args = ["--enforce-eager"]
    # Based on https://github.com/openai/openai-cookbook/blob/main/examples/Whisper_prompting_guide.ipynb.
    prompt = "THE FIRST WORDS I SPOKE"
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        transcription = await client.audio.transcriptions.create(
            model=model_name,
            file=mary_had_lamb,
            language="en",
            response_format="text",
            temperature=0.0)
        out = json.loads(transcription)['text']
        assert "Mary had a little lamb," in out
        # This should "force" whisper to continue prompt in all caps
        transcription_wprompt = await client.audio.transcriptions.create(
            model=model_name,
            file=mary_had_lamb,
            language="en",
            response_format="text",
            prompt=prompt,
            temperature=0.0)
        out_capital = json.loads(transcription_wprompt)['text']
        assert prompt not in out_capital


@pytest.mark.asyncio
async def test_bad_requests(mary_had_lamb):
    model_name = "openai/whisper-small"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        # invalid language
        with pytest.raises(openai.BadRequestError):
            await client.audio.transcriptions.create(model=model_name,
                                                     file=mary_had_lamb,
                                                     language="hh",
                                                     temperature=0.0)

        # Expect audio too long: repeat the timeseries
        mary_had_lamb.seek(0)
        audio, sr = librosa.load(mary_had_lamb)
        repeated_audio = np.tile(audio, 10)
        # Repeated audio to buffer
        buffer = io.BytesIO()
        sf.write(buffer, repeated_audio, sr, format='WAV')
        buffer.seek(0)
        with pytest.raises(openai.BadRequestError):
            await client.audio.transcriptions.create(model=model_name,
                                                     file=buffer,
                                                     language="en",
                                                     temperature=0.0)


@pytest.mark.asyncio
async def test_non_asr_model(winning_call):
    # text to text model
    model_name = "JackFram/llama-68m"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        res = await client.audio.transcriptions.create(model=model_name,
                                                       file=winning_call,
                                                       language="en",
                                                       temperature=0.0)
        assert res.code == 400 and not res.text
        assert res.message == "The model does not support Transcriptions API"


@pytest.mark.asyncio
async def test_completion_endpoints():
    # text to text model
    model_name = "openai/whisper-small"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        res = await client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "system",
                "content": "You are a helpful assistant."
            }])
        assert res.code == 400
        assert res.message == "The model does not support Chat Completions API"

        res = await client.completions.create(model=model_name, prompt="Hello")
        assert res.code == 400
        assert res.message == "The model does not support Completions API"


@pytest.mark.asyncio
async def test_streaming_response(winning_call):
    model_name = "openai/whisper-small"
    server_args = ["--enforce-eager"]
    transcription = ""
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        res_no_stream = await client.audio.transcriptions.create(
            model=model_name,
            file=winning_call,
            response_format="json",
            language="en",
            temperature=0.0)
        # Unfortunately this only works when the openai client is patched
        # to use streaming mode, not exposed in the transcription api.
        original_post = AsyncAPIClient.post

        async def post_with_stream(*args, **kwargs):
            kwargs['stream'] = True
            return await original_post(*args, **kwargs)

        with patch.object(AsyncAPIClient, "post", new=post_with_stream):
            client = remote_server.get_async_client()
            res = await client.audio.transcriptions.create(
                model=model_name,
                file=winning_call,
                language="en",
                temperature=0.0,
                extra_body=dict(stream=True))
            # Reconstruct from chunks and validate
            async for chunk in res:
                # just a chunk
                text = chunk.choices[0]['delta']['content']
                transcription += text

        assert transcription == res_no_stream.text


@pytest.mark.asyncio
async def test_stream_options(winning_call):
    model_name = "openai/whisper-small"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        original_post = AsyncAPIClient.post

        async def post_with_stream(*args, **kwargs):
            kwargs['stream'] = True
            return await original_post(*args, **kwargs)

        with patch.object(AsyncAPIClient, "post", new=post_with_stream):
            client = remote_server.get_async_client()
            res = await client.audio.transcriptions.create(
                model=model_name,
                file=winning_call,
                language="en",
                temperature=0.0,
                extra_body=dict(stream=True,
                                stream_include_usage=True,
                                stream_continuous_usage_stats=True))
            final = False
            continuous = True
            async for chunk in res:
                if not len(chunk.choices):
                    # final usage sent
                    final = True
                else:
                    continuous = continuous and hasattr(chunk, 'usage')
            assert final and continuous
