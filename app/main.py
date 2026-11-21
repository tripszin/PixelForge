from __future__ import annotations

import json
import hashlib
import secrets
import sys
import threading
import uuid
import sqlite3
import shutil
from datetime import datetime, timezone
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from safetensors import safe_open
from safetensors.torch import load_file
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor" / "pixie"))

from train.pixie32.p32_model import P32, P32Config  # noqa: E402
from train.v3_model import ClipTextPenultimate  # noqa: E402


class Settings(BaseSettings):
    pixie_model_dir: Path = ROOT / "models" / "pixie-32"
    clip_dir: Path = ROOT / "models" / "clip-vit-large-patch14"
    output_dir: Path = ROOT / "outputs" / "generated"
    cache_dir: Path = ROOT / "outputs" / "cache"
    database_path: Path = ROOT / "outputs" / "pixelforge.db"
    export_dir: Path = ROOT / "outputs" / "godot-assets"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
settings.output_dir.mkdir(parents=True, exist_ok=True)
settings.database_path.parent.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="PixelForge API", version="0.3.0")
CANDIDATE_COUNT = 2
PIPELINE_REVISION = "pixelforge-p32-v2"


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=2, max_length=500)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    scale: int = Field(default=16, ge=4, le=32)
    style_id: str | None = None

    @field_validator("prompt")
    @classmethod
    def trim_prompt(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Descreva o sprite com pelo menos 2 caracteres.")
        return value


class StyleRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    bible: str = Field(min_length=2, max_length=1000)

    @field_validator("name", "bible")
    @classmethod
    def trim_style_fields(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Preencha o nome e as regras do estilo visual.")
        return value


class PromoteRequest(BaseModel):
    image_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    prompt: str = Field(min_length=2, max_length=500)
    style_id: str | None = None
    seed: int = Field(ge=0, le=2**32 - 1)


def _db() -> sqlite3.Connection:
    connection = sqlite3.connect(settings.database_path)
    connection.row_factory = sqlite3.Row
    return connection


def _init_db() -> None:
    with _db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS styles (
                id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, bible TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY, image_id TEXT NOT NULL, prompt TEXT NOT NULL,
                style_id TEXT, seed INTEGER NOT NULL, created_at TEXT NOT NULL,
                FOREIGN KEY(style_id) REFERENCES styles(id)
            );
        """)


_init_db()


_model = None
_clip = None
_palette = None
_side = 32
_tokenizer = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model_lock = threading.Lock()
_generation_lock = threading.Lock()


def _load_model() -> None:
    global _model, _clip, _palette, _side, _tokenizer
    if _model is not None:
        return
    with _model_lock:
        if _model is not None:
            return
        model_dir, clip_dir = settings.pixie_model_dir, settings.clip_dir
        required = [model_dir / "config.json", model_dir / "model.safetensors", model_dir / "palette.npy", clip_dir / "model.safetensors", clip_dir / "tokenizer.json"]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"Arquivos do modelo ausentes: {', '.join(str(p) for p in missing)}")

        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        model_config = P32Config(**{field.name: config[field.name] for field in fields(P32Config) if field.name in config})
        model = P32(model_config)
        model.load_state_dict(load_file(str(model_dir / "model.safetensors"), device="cpu"), strict=True)

        clip = ClipTextPenultimate()
        # The CLIP file also contains a vision tower that this sprite model does not use.
        clip_keys = ["text_model.embeddings.token_embedding.weight", "text_model.embeddings.position_embedding.weight"]
        for i in range(len(clip.layers)):
            prefix = f"text_model.encoder.layers.{i}."
            for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "layer_norm1", "layer_norm2", "mlp.fc1", "mlp.fc2"):
                clip_keys.extend((prefix + name + ".weight", prefix + name + ".bias"))
        with safe_open(str(clip_dir / "model.safetensors"), framework="pt", device="cpu") as source:
            clip_state = {key: source.get_tensor(key) for key in clip_keys}
        clip.load_hf(clip_state)
        del clip_state

        _model = model.to(_device).eval()
        _clip = clip.to(_device).eval()
        _palette = np.load(model_dir / config.get("palette_file", "palette.npy"))
        _side = int(config.get("image_size", 32))
        _tokenizer = Tokenizer.from_file(str(clip_dir / "tokenizer.json"))


def _prompt_ids(prompt: str) -> torch.Tensor:
    assert _tokenizer is not None
    content = _tokenizer.encode(prompt, add_special_tokens=False).ids[:16]
    row = np.full(18, 49407, dtype=np.int64)
    row[0] = 49406
    row[1:1 + len(content)] = content
    return torch.as_tensor(row[None, :], dtype=torch.long, device=_device)


def _empty_ids() -> torch.Tensor:
    row = torch.full((1, 18), 49407, dtype=torch.long, device=_device)
    row[:, 0] = 49406
    return row


def _cache_key(prompt: str, base_seed: int, scale: int) -> str:
    identity = {
        "pipeline": PIPELINE_REVISION,
        "pixie_model": "pixie-32-1a26888728f3c523",
        "clip_model": "clip-vit-large-patch14-local",
        "prompt": prompt,
        "base_seed": base_seed,
        "scale": scale,
        "candidate_count": CANDIDATE_COUNT,
        "sampler": {"cfg": 2.0, "temperature": 0.8, "top_k": 32},
    }
    canonical = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_cached_result(cache_key: str) -> dict | None:
    manifest_path = settings.cache_dir / f"{cache_key}.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidates = manifest["candidates"]
        if manifest.get("cache_key") != cache_key or len(candidates) != CANDIDATE_COUNT:
            return None
        for candidate in candidates:
            image_id = candidate["id"]
            if len(image_id) != 32 or any(char not in "0123456789abcdef" for char in image_id):
                return None
            if not (settings.output_dir / f"{image_id}.png").is_file():
                return None
        return {"base_seed": manifest["base_seed"], "candidates": candidates, "cache_hit": True}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cached_result(cache_key: str, result: dict) -> None:
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"cache_key": cache_key, "base_seed": result["base_seed"], "candidates": result["candidates"]}
    destination = settings.cache_dir / f"{cache_key}.json"
    temporary = settings.cache_dir / f".{cache_key}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(destination)


@app.get("/api/health")
def health() -> dict:
    required = [
        settings.pixie_model_dir / "config.json", settings.pixie_model_dir / "model.safetensors",
        settings.pixie_model_dir / "palette.npy", settings.clip_dir / "model.safetensors",
        settings.clip_dir / "tokenizer.json",
    ]
    ready = all(path.is_file() for path in required)
    device_name = torch.cuda.get_device_name(0) if _device.type == "cuda" else "CPU"
    return {"status": "ok", "engine": "ready" if ready else "models_missing", "device": device_name}


@app.get("/api/styles")
def list_styles() -> list[dict]:
    with _db() as db:
        return [dict(row) for row in db.execute("SELECT id, name, bible, created_at FROM styles ORDER BY name COLLATE NOCASE")]


@app.post("/api/styles")
def create_style(payload: StyleRequest) -> dict:
    style_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        with _db() as db:
            db.execute("INSERT INTO styles (id, name, bible, created_at) VALUES (?, ?, ?, ?)",
                       (style_id, payload.name.strip(), payload.bible.strip(), created_at))
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Já existe um estilo com esse nome.") from exc
    return {"id": style_id, "name": payload.name.strip(), "bible": payload.bible.strip(), "created_at": created_at}


@app.get("/api/assets")
def list_assets() -> list[dict]:
    with _db() as db:
        return [dict(row) for row in db.execute("""
            SELECT a.id, a.image_id, a.prompt, a.style_id, a.seed, a.created_at,
                   s.name AS style_name
            FROM assets a LEFT JOIN styles s ON s.id = a.style_id
            ORDER BY a.created_at DESC
        """)]


@app.post("/api/assets")
def promote_asset(payload: PromoteRequest) -> dict:
    image_path = settings.output_dir / f"{payload.image_id}.png"
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Imagem não encontrada.")
    if payload.style_id:
        with _db() as db:
            if db.execute("SELECT 1 FROM styles WHERE id = ?", (payload.style_id,)).fetchone() is None:
                raise HTTPException(status_code=404, detail="Estilo não encontrado.")
    asset_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    with _db() as db:
        db.execute("INSERT INTO assets (id, image_id, prompt, style_id, seed, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                   (asset_id, payload.image_id, payload.prompt.strip(), payload.style_id, payload.seed, created_at))
    return {"id": asset_id, "image_id": payload.image_id, "prompt": payload.prompt.strip(),
            "style_id": payload.style_id, "seed": payload.seed, "created_at": created_at}


@app.post("/api/export/godot")
def export_godot() -> dict:
    settings.export_dir.mkdir(parents=True, exist_ok=True)
    with _db() as db:
        assets = [dict(row) for row in db.execute("""
            SELECT a.id, a.image_id, a.prompt, a.seed, a.created_at, s.name AS style_name
            FROM assets a LEFT JOIN styles s ON s.id = a.style_id ORDER BY a.created_at
        """)]
    manifest = []
    used_names: set[str] = set()
    for asset in assets:
        source = settings.output_dir / f"{asset['image_id']}.png"
        if not source.is_file():
            continue
        stem = "".join(ch.lower() if ch.isalnum() else "_" for ch in asset["prompt"][:48]).strip("_") or "sprite"
        name = stem
        suffix = 2
        while name in used_names:
            name = f"{stem}_{suffix}"
            suffix += 1
        used_names.add(name)
        filename = f"{name}.png"
        shutil.copy2(source, settings.export_dir / filename)
        manifest.append({**asset, "file": filename})
    (settings.export_dir / "pixelforge-assets.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"count": len(manifest), "path": str(settings.export_dir)}


@app.post("/api/generate")
def generate(payload: GenerateRequest) -> dict:
    prompt = payload.prompt.strip()
    style_bible = ""
    if payload.style_id:
        with _db() as db:
            style = db.execute("SELECT bible FROM styles WHERE id = ?", (payload.style_id,)).fetchone()
        if style is None:
            raise HTTPException(status_code=404, detail="Estilo não encontrado.")
        style_bible = style["bible"].strip()
    effective_prompt = f"{prompt}, {style_bible}" if style_bible else prompt
    base_seed = payload.seed if payload.seed is not None else secrets.randbelow(2**32)
    cache_key = _cache_key(effective_prompt, base_seed, payload.scale)
    try:
        with _generation_lock:
            cached = _read_cached_result(cache_key)
            if cached is not None:
                return cached
            _load_model()

            from PIL import Image

            settings.output_dir.mkdir(parents=True, exist_ok=True)
            candidates = []
            with torch.inference_mode():
                text = _prompt_ids(effective_prompt)
                clip_text = _clip(text)
                clip_empty = _clip(_empty_ids())
                for index in range(CANDIDATE_COUNT):
                    seed = (base_seed + index) % (2**32)
                    generator = torch.Generator(device=_device).manual_seed(seed)
                    samples = _model.sample(clip_text, clip_empty, cfg=2.0, temperature=0.8, top_k=32, generator=generator)
                    indices = samples[0].reshape(_side, _side).cpu().numpy()
                    rgb = _palette[indices][..., :3].astype(np.uint8)
                    alpha = ((indices != 0) * 255).astype(np.uint8)[..., None]
                    sprite = Image.fromarray(np.concatenate((rgb, alpha), axis=-1), "RGBA")
                    image = sprite.resize((_side * payload.scale, _side * payload.scale), Image.Resampling.NEAREST)
                    image_id = uuid.uuid4().hex
                    image_path = settings.output_dir / f"{image_id}.png"
                    temporary_path = settings.output_dir / f".{image_id}.tmp"
                    image.save(temporary_path, format="PNG", optimize=True)
                    temporary_path.replace(image_path)
                    candidates.append({
                        "id": image_id,
                        "url": f"/api/images/{image_id}",
                        "seed": seed,
                        "width": image.width,
                        "height": image.height,
                        "native_width": _side,
                        "native_height": _side,
                    })
            result = {"base_seed": base_seed, "cache_hit": False, "candidates": candidates}
            _write_cached_result(cache_key, result)
            return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha na geração local: {exc}") from exc


@app.get("/api/images/{image_id}")
def get_image(image_id: str) -> FileResponse:
    if len(image_id) != 32 or any(ch not in "0123456789abcdef" for ch in image_id):
        raise HTTPException(status_code=404, detail="Imagem não encontrada.")
    path = settings.output_dir / f"{image_id}.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Imagem não encontrada.")
    return FileResponse(path, media_type="image/png")


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
