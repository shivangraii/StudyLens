"""Video/audio -> timestamped transcript segments.

Strategy for YouTube:
  1. Try the official captions (fast, no GPU, no download).
  2. Fall back to downloading audio with yt-dlp and transcribing with faster-whisper.
Local audio/video files always go through faster-whisper.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from ..config import Config
from ..schemas import youtube_id

_whisper = None


def _fetch_captions(video_id: str, langs=("en", "hi")) -> list[dict]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return []
    try:
        if hasattr(YouTubeTranscriptApi, "get_transcript"):  # library < 1.0
            raw = YouTubeTranscriptApi.get_transcript(video_id, languages=list(langs))
            items = [(r["text"], r["start"], r["duration"]) for r in raw]
        else:  # library >= 1.0
            fetched = YouTubeTranscriptApi().fetch(video_id, languages=list(langs))
            items = [(s.text, s.start, s.duration) for s in fetched]
    except Exception:
        return []
    return [
        {"text": t.replace("\n", " ").strip(), "start": float(s), "end": float(s) + float(d)}
        for t, s, d in items
        if t.strip()
    ]


def transcribe(path: str | Path, cfg: Config) -> list[dict]:
    """faster-whisper transcription -> [{"text","start","end"}]."""
    global _whisper
    from faster_whisper import WhisperModel

    if _whisper is None:
        _whisper = WhisperModel(cfg.whisper_model, device=cfg.device, compute_type=cfg.whisper_compute)
    segments, _info = _whisper.transcribe(str(path), vad_filter=True, beam_size=1)
    return [{"text": s.text.strip(), "start": float(s.start), "end": float(s.end)} for s in segments if s.text.strip()]


def parse_youtube(url: str, cfg: Config) -> tuple[str, list[dict], str]:
    """Return (title, segments, how) where how is 'captions' or 'whisper'."""
    vid = youtube_id(url)
    if not vid:
        raise ValueError("That doesn't look like a YouTube URL.")
    title = f"YouTube {vid}"
    try:
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True, "noplaylist": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title") or title
    except Exception:
        pass

    segments = _fetch_captions(vid)
    if segments:
        return title, segments, "captions"

    import yt_dlp

    with tempfile.TemporaryDirectory() as tmp:
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(Path(tmp) / "%(id)s.%(ext)s"),
            "quiet": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = ydl.prepare_filename(info)
        segments = transcribe(audio_path, cfg)
    if not segments:
        raise ValueError("Transcription produced no text.")
    return title, segments, "whisper"
