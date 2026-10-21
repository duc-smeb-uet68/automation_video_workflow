from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Config(Record):
    jobs_dir: Path = Path("jobs")
    input_mode: Literal["script", "topic"] = "script"
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    text_provider: Literal["openrouter", "openai"] = "openrouter"
    openrouter_model: str = Field(default="openai/gpt-oss-120b", min_length=1)
    openai_model: str = "gpt-4.1"
    image_provider: Literal["openrouter", "gemini"] = "openrouter"
    openrouter_image_model: str = Field(default="google/gemini-3.1-flash-lite-image", min_length=1)
    gemini_model: str = "gemini-3.1-flash-lite-image"
    elevenlabs_model: str = "eleven_multilingual_v2"
    voice_id: str = ""
    voice_settings: dict = Field(default_factory=lambda: {
        "stability": 0.71, "similarity_boost": 0.75, "style": 0.37,
        "use_speaker_boost": True, "speed": 0.95,
    })
    width: int = Field(default=1920, ge=320)
    height: int = Field(default=1080, ge=180)
    fps: int = Field(default=30, ge=1, le=60)
    crf: int = Field(default=20, ge=0, le=40)
    encoder_preset: str = "veryfast"
    threads: int = Field(default=4, ge=1, le=32)
    max_attempts: int = Field(default=3, ge=1, le=5)
    retry_backoff_seconds: float = Field(default=2, ge=0, le=30)
    request_timeout_seconds: float = Field(default=180, ge=1, le=600)
    image_interval_seconds: float = Field(default=1, ge=0, le=60)
    music_path: Path | None = None
    font_path: Path | None = None
    character_reference: Path | None = Field(default=None, description="Accepted for legacy configuration; ignored by text-only generation.")
    max_silence_seconds: float = Field(default=3, gt=0)
    spending_ceiling_usd: float | None = Field(default=None, gt=0)
    request_cost_caps_usd: dict[str, float] = Field(default_factory=dict)

    @property
    def text_model(self) -> str:
        return self.openrouter_model if self.text_provider == "openrouter" else self.openai_model

    @property
    def text_key_env(self) -> str:
        return "OPENROUTER_API_KEY" if self.text_provider == "openrouter" else "OPENAI_API_KEY"

    @property
    def image_model(self) -> str:
        return self.openrouter_image_model if self.image_provider == "openrouter" else self.gemini_model

    @property
    def image_key_env(self) -> str:
        return "OPENROUTER_API_KEY" if self.image_provider == "openrouter" else "GEMINI_API_KEY"

    @property
    def generation_keys(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.text_key_env, "ELEVENLABS_API_KEY", self.image_key_env)))

    @property
    def text_cache_settings(self) -> dict:
        return {"provider": self.text_provider, "model": self.text_model, "adapter_version": 1}

    @property
    def image_cache_settings(self) -> dict:
        settings = {"provider": self.image_provider, "model": self.image_model, "adapter_version": 2,
                    "strategy": "text_only", "resolution": "1K", "aspect_ratio": "16:9"}
        if self.image_provider == "openrouter":
            settings["gemini_fallback_model"] = self.gemini_model
        return settings

    @model_validator(mode="after")
    def frame_shape(self):
        if self.width % 2 or self.height % 2 or abs(self.width / self.height - 16 / 9) > 0.01:
            raise ValueError("Output must have even dimensions and a 16:9 aspect ratio")
        if any(cost < 0 for cost in self.request_cost_caps_usd.values()):
            raise ValueError("Request cost caps cannot be negative")
        return self

    @classmethod
    def load(cls, path: Path | None = None):
        data = json.loads(path.read_text(encoding="utf-8-sig")) if path else {}
        config = cls.model_validate(data)
        base = path.resolve().parent if path else Path.cwd()
        for name in ("jobs_dir", "music_path", "font_path", "character_reference"):
            value = getattr(config, name)
            if value is not None and not value.is_absolute():
                setattr(config, name, (base / value).resolve())
        # Reuse the installed Kdenlive distribution if FFmpeg is not on PATH.
        for name in ("ffmpeg", "ffprobe"):
            value = getattr(config, name)
            candidate = Path("D:/kenlive/bin") / f"{name}.exe"
            if not shutil.which(value) and value == name and candidate.is_file():
                setattr(config, name, str(candidate))
        return config


class Sentence(Record):
    id: str
    paragraph_id: str
    text: str
    char_start: int
    char_end: int


class Script(Record):
    narration: str
    character_count: int
    sentences: list[Sentence]
    source_references: list[str] = Field(default_factory=list)


class Word(Record):
    text: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    char_start: int
    char_end: int
    sentence_id: str

    @model_validator(mode="after")
    def ordered(self):
        if self.end < self.start or self.char_end <= self.char_start:
            raise ValueError("Invalid word interval")
        return self


class Cue(Record):
    id: str
    text: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    word_start: int = Field(ge=0)
    word_end: int = Field(gt=0)
    scene_group: str = ""
    character_ids: list[str] = Field(default_factory=list)
    scene_description: str = ""
    overlay_text: str | None = None
    motion: Literal["hold", "slow_push"] = "hold"
    image_path: str | None = None
    overlay_path: str | None = None
    start_frame: int = 0
    end_frame: int = 0


class Stage(Record):
    status: Literal["running", "completed", "failed"] = "running"
    input_hash: str
    attempts: int = 0
    errors: list[str] = Field(default_factory=list)
    result: dict = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)


class Manifest(Record):
    schema_version: int = 1
    job_id: str
    topic: str
    mode: Literal["production", "script", "fixture", "imported"] = "production"
    image_strategy: Literal["reference", "text_only"] = "reference"
    script_input_hash: str | None = None
    status: Literal["pending", "running", "failed", "completed", "interrupted"] = "pending"
    script_character_count: int = 0
    fps: int = 30
    audio_path: str | None = None
    audio_duration: float = 0
    total_frames: int = 0
    cues: list[Cue] = Field(default_factory=list)
    stages: dict[str, Stage] = Field(default_factory=dict)
    usage: list[dict] = Field(default_factory=list)
    error: str | None = None
