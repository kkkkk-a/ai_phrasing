import os
import json
import random
import time
import re
import asyncio
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
# ---------------------------------------------------------
# 0. ロギング設定 (詳細デバッグ・ファイル出力)
# ---------------------------------------------------------
# ★追加: 本番環境(Render)かどうかの判定フラグ
IS_PRODUCTION = os.getenv("RENDER") is not None

log_format = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

# コンソール出力 (常に有効)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_format)

logger = logging.getLogger("RAI_CORE")
logger.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

# ★修正: 本番環境以外(ローカル)の場合のみファイルに出力する
if not IS_PRODUCTION:
    file_handler = RotatingFileHandler("debug.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
else:
    logger.info("Production mode detected: File logging disabled.")

# 外部ライブラリのログを抑制
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.INFO)

app = FastAPI(title="RAI - Relationship Drive Interface", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. 型定義 & 設定管理
# ---------------------------------------------------------

class ScenarioConfig(BaseModel):
    char: str
    rel: str
    sit: str
    me: str = "私"
    you: str = "あなた"

class InteractionRequest(BaseModel):
    sliders: Dict[str, float]
    scenario: ScenarioConfig
    image_data: Optional[str] = None
    imported_text: Optional[str] = None 

class SettingsManager:
    def __init__(self):
        self.config = {}
        self.reload()

    def reload(self):
        try:
            if os.path.exists("settings.json"):
                with open("settings.json", "r", encoding="utf-8") as f:
                    self.config = json.load(f)
                logger.debug(f"Settings reloaded: {self.config.get('mode', 'unknown')} mode")
            else:
                self.config = {}
        except Exception as e:
            logger.error(f"Settings Load Error: {e}")

    def get_api_key(self) -> str:
        env_key = os.getenv("OPENROUTER_API_KEY")
        if env_key: return env_key
        return self.config.get("api_key", "")

    def get_system_prompt(self) -> str:
        return self.config.get("system_prompt", "")

settings_manager = SettingsManager()

# ---------------------------------------------------------
# 2. 履歴・コンテキスト管理
# ---------------------------------------------------------

class HistoryManager:
    def __init__(self):
        self.file_path = "history.json"
        self.lock = asyncio.Lock()
        self.memory_history = [] # ★追加: 本番環境用のメモリ保存領域

    async def load(self) -> List[Dict]:
        # ★追加: 本番環境ならメモリから返す
        if IS_PRODUCTION:
            return self.memory_history

        async with self.lock:
            if not os.path.exists(self.file_path): return []
            try:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._read_file)
            except Exception as e:
                logger.error(f"History Read Error: {e}")
                return []

    def _read_file(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def save(self, data: List[Dict]):
        # ★追加: 本番環境ならメモリに保存して終了
        if IS_PRODUCTION:
            self.memory_history = data
            return

        async with self.lock:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._write_file, data)
            except Exception as e:
                logger.error(f"History Write Error: {e}")

    def _write_file(self, data):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    async def append_log(self, entry: Dict):
        history = await self.load()
        history.insert(0, entry)
        limit = settings_manager.config.get("log_limit", 100)
        await self.save(history[:limit])

history_manager = HistoryManager()

# ---------------------------------------------------------
# 3. モデル管理 (無料モデル自動スキャン)
# ---------------------------------------------------------

class ModelManager:
    def __init__(self):
        self.available_models = []
        self.blacklist = set()
        self.last_update = 0

    async def refresh_models(self):
        if time.time() - self.last_update < 600 and self.available_models:
            return
        logger.debug("Scanning OpenRouter for free models...")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get("https://openrouter.ai/api/v1/models")
                if resp.status_code == 200:
                    all_models = resp.json().get("data", [])
                    free_models = [m["id"] for m in all_models if ":free" in m["id"]]
                    
                    # 特定のモデルを最優先し、次にGemini、その後にその他を並べる
                    priority_model = "arcee-ai/trinity-large-preview:free"
                    
                    target = [m for m in free_models if m == priority_model]
                    geminis = [m for m in free_models if "google/gemini" in m and m != priority_model]
                    others = [m for m in free_models if "google/gemini" not in m and m != priority_model]
                    
                    self.available_models = target + geminis + others
                    self.blacklist = set()
                    logger.info(f"Model scan complete. Priority: {priority_model}")
        except Exception as e:
            logger.error(f"Model scan failed: {e}")
        self.last_update = time.time()

    def get_candidates(self):
        return [m for m in self.available_models if m not in self.blacklist]

    def report_error(self, model_id):
        logger.warning(f"Model blacklisted: {model_id}")
        self.blacklist.add(model_id)

model_manager = ModelManager()

# ---------------------------------------------------------
# 4. エンジン (生成 & 人称置換)
# ---------------------------------------------------------
class Engine:
    def __init__(self):
        self.dictionary = {}
        self.pollution_dict = {}
        self.pollution_lock = asyncio.Lock()
        self.dictionary_writable = True
        self.load_dictionary()
        self.load_pollution_dict()

    def load_dictionary(self):
        """辞書ファイル(dictionary.json)を読み込む"""
        if os.path.exists("dictionary.json"):
            try:
                with open("dictionary.json", "r", encoding="utf-8") as f:
                    self.dictionary = json.load(f)
            except Exception as e:
                logger.error(f"Dictionary Load Error: {e}")
                
    def load_pollution_dict(self):
        """汚染辞書の読み込み（破壊防止機能付き）"""
        path = "pollution_dict.json"
        if not os.path.exists(path):
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({}, f)
                self.pollution_dict = {}
                self.dictionary_writable = True
            except Exception as e:
                logger.error(f"Init Error: {e}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    self.pollution_dict = json.loads(content)
                else:
                    self.pollution_dict = {}
            self.dictionary_writable = True
            logger.info(f"Pollution dict loaded. {len(self.pollution_dict)} words.")
        except Exception as e:
            logger.critical(f"FATAL: pollution_dict.json Load Error: {e}")
            self.dictionary_writable = False
            self.pollution_dict = {}
            import shutil
            if os.path.exists(path):
                shutil.copy(path, path + ".bak")

    async def _update_pollution_dict(self, text: str):
        """汚染辞書の自動更新（あらゆる異物を検知し、ファイルを保護する）"""
        if IS_PRODUCTION: 
            return
        if not text or not getattr(self, 'dictionary_writable', True): return
        
        safe_pattern = r'[ぁ-んァ-ン一-龥、。！？「」『』（）…ー―　\n\r0-9０-９〜～・]'
        found_pollutants = re.findall(r'[a-zA-Z]{2,}|\{\{.*?\}\}|parameter\s*:', text)
        
        for char in text:
            if not re.match(safe_pattern, char) and not char.isspace() and not re.match(r'[a-zA-Z]', char):
                found_pollutants.append(char)
        
        unique_pollutants = list(set(found_pollutants))
        new_words_found = False
        for word in unique_pollutants:
            if word not in self.pollution_dict:
                self.pollution_dict[word] = ""
                logger.info(f"★新規汚染検知: '{word}'")
                new_words_found = True
        
        # 本番環境(RENDER)でなければ保存する
        if new_words_found and not IS_PRODUCTION:
            try:
                async with self.pollution_lock:
                    temp_path = "pollution_dict.json.tmp"
                    with open(temp_path, "w", encoding="utf-8") as f:
                        json.dump(self.pollution_dict, f, indent=2, ensure_ascii=False)
                    os.replace(temp_path, "pollution_dict.json")
            except Exception as e:
                logger.error(f"Pollution dictionary save error: {e}")

    def _rule_based_translate(self, text: str) -> str:
        """汚染辞書を使って浄化する"""
        text = re.sub(r'\(.*?\)|評価:|要件|憲法|順守|安全基準|parameter\s*:', '', text)
        text = re.sub(r'[가-힣]|[а-яА-Я]', '', text)
        if self.pollution_dict:
            sorted_keys = sorted(self.pollution_dict.keys(), key=len, reverse=True)
            for key in sorted_keys:
                trans = self.pollution_dict[key]
                pattern = r'\b' + re.escape(key) + r'\b' if len(key) > 1 else re.escape(key)
                text = re.sub(pattern, trans, text, flags=re.IGNORECASE)
        return text.strip()

    def _trim_incomplete_sentence(self, text: str) -> str:
        """ぶつ切り防止：末尾が適切な句読点で終わっていない場合、最後の一文を削る"""
        if not text: return ""
        valid_ends = ('。', '！', '？', '」', '』', '）', '”', '…')
        
        if text.endswith(valid_ends):
            return text
        
        last_punct = -1
        for p in valid_ends:
            pos = text.rfind(p)
            if pos > last_punct:
                last_punct = pos
        
        if last_punct != -1:
            return text[:last_punct + 1]
        return text

    # _is_garbage はもう判定には使いませんが、ログ出力用に残します
    def _is_garbage(self, text: str) -> bool:
        if not text: return True
        allowed_pattern = r'[ぁ-んァ-ン一-龥、。！？「」『』（）…ー―　\n\r0-9０-９〜～・]'
        allowed_chars_count = len(re.findall(allowed_pattern, text))
        if len(text) > 0 and allowed_chars_count < len(text):
            return True
        return False

    def _needs_translation(self, text: str) -> bool:
        clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return bool(re.search(r'[a-zA-Z]{2,}', clean))

    async def _force_translate(self, text: str, api_key: str, model_id: str) -> str:
        async with httpx.AsyncClient(timeout=50.0) as client:
            try:
                ms = [{"role": "system", "content": "小説の翻訳校閲者です。英語やノイズを排除し、情緒的な日本語のみにリライトせよ。翻訳結果のみを出力すること。"}, {"role": "user", "content": f"浄化：\n\n{text}"}]
                resp = await client.post("https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": model_id, "messages": ms, "temperature": 0.2})
                if resp.status_code == 200:
                    res = resp.json()['choices'][0]['message']['content'].strip()
                    # 翻訳結果が空っぽなら元のテキストを返す
                    return res if res else ""
            except: pass
        return ""

    def _get_dictionary_seeds(self, sliders: Dict[str, float]) -> str:
        dist, temp, purity, habit = sliders.get("dist", 0.5), sliders.get("temp", 0.5), sliders.get("purity", 0.5), sliders.get("habituation", 0.5)
        seeds = []
        try:
            if dist > 0.7: seeds.extend(random.sample(self.dictionary.get("distance", {}).get("close", []), 2))
            elif dist < 0.3: seeds.extend(random.sample(self.dictionary.get("distance", {}).get("far", []), 2))
            if temp > 0.7:
                if purity > 0.7: seeds.extend(random.sample(self.dictionary.get("purity", {}).get("innocent_heat", []), 2))
                elif habit > 0.7: seeds.extend(random.sample(self.dictionary.get("habituation", {}).get("routine", []), 1))
                else: seeds.extend(random.sample(self.dictionary.get("temperature", {}).get("hot", []), 2))
            seeds.append(random.choice(self.dictionary.get("physical_details", {}).get("breath", ["..."])))
        except: return "……。"
        return "\n・".join(seeds)

    def _apply_pronouns(self, text: str, scenario: ScenarioConfig) -> str:
        if not text: return ""
        return text.replace("{{i}}", scenario.me).replace("{{me}}", scenario.me).replace("{{you}}", scenario.you)

    def _get_static_text(self, sliders: Dict[str, float], scenario: ScenarioConfig) -> str:
        dist, temp, length, style = sliders.get("dist", 0.5), sliders.get("temp", 0.5), sliders.get("length", 0.5), sliders.get("style", 0.5)
        asym, purity, habit = sliders.get("asymmetry", 0.5), sliders.get("purity", 0.5), sliders.get("habituation", 0.5)
        is_dominant = asym < 0.4
        parts = []
        try:
            parts.append(random.choice(self.dictionary.get("distance", {}).get("close" if dist > 0.6 else "far", ["..."])))
            if length > 0.2:
                if purity > 0.6 and temp > 0.5: parts.append(random.choice(self.dictionary.get("purity", {}).get("innocent_heat", ["..."])))
                else: parts.append(random.choice(self.dictionary.get("temperature", {}).get("hot" if temp > 0.6 else "cold", ["..."])))
            if length > 0.4: parts.append(random.choice(self.dictionary.get("physical_details", {}).get("touch" if style > 0.5 else "eyes", ["..."])))
            if length > 0.5: parts.append(random.choice(self.dictionary.get("asymmetry", {}).get("dominant" if is_dominant else "submissive", ["..."])))
            if length > 0.6 and habit > 0.7: parts.append(random.choice(self.dictionary.get("habituation", {}).get("routine", ["..."])))
            if length > 0.8: parts.append(random.choice(self.dictionary.get("metaphors", {}).get("pleasure" if temp > 0.5 else "control", ["..."])))
            parts.append(random.choice(self.dictionary.get("physical_details", {}).get("breath", ["..."])))
        except: parts = ["……。"]
        return self._apply_pronouns(" ".join(parts), scenario)

    def _clean_text(self, text: str) -> str:
        if not text: return ""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'^(Sure|Okay|Here is|小説の続き).*?(\n|：|:)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _build_system_prompt(self, req: InteractionRequest) -> str:
        s = req.sliders
        conf = settings_manager.config
        l_val = s.get('length', 0.5)
        if l_val < 0.2: len_inst = "極めて簡潔に(100字以内)。"
        elif l_val < 0.4: len_inst = "短文(300字程度)。"
        elif l_val < 0.7: len_inst = "標準的な長さ(500字前後)。"
        elif l_val < 0.9: len_inst = "濃密な長文(1000字程度)。"
        else: len_inst = "極めて執拗な長文(2000字以上)で、文章の途中でトークン制限に達しないよう、指定文字数内で必ず物語を完結させよ。絶対に文章を途中で終わらせるな。"

        st_val = s.get('style', 0.5)
        style_inst = "肉感的・生理的" if st_val > 0.6 else "情緒的・比喩的"
        seeds = self._get_dictionary_seeds(s)
        role = conf.get("role_settings", {}).get("dominant_label", "支配者") if s.get('asymmetry', 0.5) < 0.4 else conf.get("role_settings", {}).get("submissive_label", "従属者")

        base = conf.get("system_prompt", "")
        reps = {"{{char}}": req.scenario.char, "{{rel}}": req.scenario.rel, "{{sit}}": req.scenario.sit, "{{me}}": req.scenario.me, "{{you}}": req.scenario.you, "{{role}}": role, "{{length}}": len_inst, "{{style}}": style_inst, "{{dist}}": f"{s.get('dist', 0.5):.2f}", "{{temp}}": f"{s.get('temp', 0.5):.2f}", "{{purity}}": f"{s.get('purity', 0.5):.2f}", "{{habituation}}": f"{s.get('habituation', 0.5):.2f}"}
        for k, v in reps.items(): base = base.replace(k, str(v))
        
        base += f"\n\n[シード]:\n・{seeds}\n\n[文体]: {style_inst}\n[ノルマ]: {len_inst}\n\n[重要]:日本語のみ。英語・記号禁止。必ず句読点で文章を完結させよ。"
        return base

    async def generate(self, req: InteractionRequest, user_key:str=None, user_model:str=None, provider:str="openrouter", custom_url:str=None) -> Tuple[str, str, Dict]:
        """テキスト生成のメインフロー（デバッグ強化・判定緩和版）"""
        
        # 1. API設定
        api_key = user_key if user_key else settings_manager.get_api_key()
        
        # URL決定
        if provider == "openai": base_url = "https://api.openai.com/v1/chat/completions"
        elif provider == "google": base_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        elif provider == "groq": base_url = "https://api.groq.com/openai/v1/chat/completions"
        elif provider == "custom" and custom_url: base_url = custom_url
        else: base_url = "https://openrouter.ai/api/v1/chat/completions"

        # 2. モデル候補の決定
        candidates = []
        # 画像認識に強いモデルのフォールバックリスト
        vision_fallbacks = [
            "google/gemini-2.0-flash-lite-preview-02-05:free",
            "google/gemini-2.0-pro-exp-02-05:free",
            "google/gemini-2.0-flash-exp:free"
        ]

        if user_model and user_model.strip():
            candidates = [user_model.strip()]
            if req.image_data:
                candidates += vision_fallbacks
        elif req.image_data:
            candidates = vision_fallbacks
        elif provider == "openrouter":
            await model_manager.refresh_models()
            candidates = model_manager.get_candidates()[:3]
        else:
            return self._get_static_text(req.sliders, req.scenario), "Error: Model Name Required", {}

        # 3. プロンプト構築
        system_prompt = self._build_system_prompt(req)
        history = await history_manager.load()
        messages = [{"role": "system", "content": system_prompt}]
        
        history_limit = 3 if req.sliders.get("length", 0.5) > 0.8 else 5
        for log in reversed(history[:history_limit]):
            if log.get("text"): messages.append({"role": "assistant", "content": log["text"]})
        
        base_msg = "続きを描写せよ。日本語のみ。必ず完結させろ。"
        if req.imported_text and req.imported_text.strip():
            user_content = f"【参照テキスト】\n{req.imported_text}\n\n{base_msg}"
        else:
            user_content = base_msg

        if req.image_data:
            messages.append({"role": "user", "content": [{"type": "text", "text": user_content}, {"type": "image_url", "image_url": {"url": req.image_data}}]})
        else:
            messages.append({"role": "user", "content": user_content})

        # 4. パラメータ
        input_temp = req.sliders.get("temp", 0.5)
        safe_temp = 0.2 + (input_temp * 0.6)
        max_tokens_val = 3000 if req.sliders.get("length", 0.5) > 0.8 else 1200

        # 5. 実行ループ
        async with httpx.AsyncClient(timeout=100.0) as client:
            for model_id in candidates:
                try:
                    logger.debug(f"Connecting to {base_url} with model {model_id}...")
                    resp = await client.post(
                        base_url, 
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "HTTP-Referer": "http://localhost:8000",
                            "X-Title": "RAI Interface",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": model_id,
                            "messages": messages,
                            "temperature": safe_temp,
                            "top_p": 0.9,
                            "repetition_penalty": 1.1,
                            "max_tokens": max_tokens_val
                        }
                    )
                    
                    if resp.status_code == 200:
                        raw = resp.json()['choices'][0]['message'].get('content', '')
                        
                        if not raw or not raw.strip(): 
                            logger.warning(f"Model {model_id} returned EMPTY content. Skipping.")
                            continue

                        # 学習
                        await self._update_pollution_dict(raw)
                        
                        # 浄化フロー
                        cleaned = self._clean_text(raw)
                        cleaned = self._rule_based_translate(cleaned)
                        
                        if self._needs_translation(cleaned):
                            translated = await self._force_translate(cleaned, api_key, model_id)
                            if translated: cleaned = translated
                        
                        cleaned = self._apply_pronouns(cleaned, req.scenario)
                        
                        # 物理削除とぶつ切り防止
                        cleaned = re.sub(r'[a-zA-Zａ-ｚＡ-Ｚ가-힣а-яА-Я\{\}]+', '', cleaned)
                        cleaned = re.sub(r'parameter\s*:\s*.*?\d+\.\d+|値を入力', '', cleaned)
                        cleaned = self._trim_incomplete_sentence(cleaned)

                        # --- ★修正点: 文字数チェックの緩和 ---
                        # 日本語（漢字・かな）が「1文字」でもあれば採用する
                        jp_count = len(re.findall(r'[ぁ-んァ-ン一-龥]', cleaned))
                        if jp_count < 1:
                            logger.warning(f"Result from {model_id} deleted because it had NO Japanese: '{cleaned}'")
                            continue 
                        
                        # ここまで来たら採用
                        return cleaned.strip(), model_id, resp.json().get('usage', {})
                    
                    else:
                        # APIエラーの詳細をログに出す
                        logger.error(f"API Error {model_id}: Status {resp.status_code} | Response: {resp.text}")
                        if provider == "openrouter": model_manager.report_error(model_id)

                except Exception as e:
                    logger.error(f"Exception error with {model_id}: {e}")
                    if provider == "openrouter": model_manager.report_error(model_id)

        return self._get_static_text(req.sliders, req.scenario) + "\n(※通信不安定)", "FilterSystem", {}

engine = Engine()

# ---------------------------------------------------------
# 5. エンドポイント
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "index.html not found"

@app.post("/api/interact")
async def interact(req: InteractionRequest, request: Request):
    settings_manager.reload()
    
    # ヘッダーからユーザー設定を取得
    user_key = request.headers.get("X-User-API-Key")
    user_model = request.headers.get("X-User-Model")
    provider = request.headers.get("X-User-Provider", "openrouter")
    custom_url = request.headers.get("X-Custom-Url")
    
    # Engineに全て渡す
    text, model, usage = await engine.generate(req, 
                                               user_key=user_key, 
                                               user_model=user_model, 
                                               provider=provider, 
                                               custom_url=custom_url)
    
    await history_manager.append_log({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "role": "assistant",
        "text": text,
        "model": model,
        "sliders": req.sliders,
        "usage": usage
    })
    
    history = await history_manager.load()
    return {"text": text, "model": model, "usage": usage, "log_count": len(history)}

@app.get("/api/history")
async def get_history():
    h = await history_manager.load()
    return {"history": h, "count": len(h)}

@app.post("/api/reset")
async def reset():
    await history_manager.save([])
    logger.info("History fully reset by user.")
    return {"status": "ok"}

@app.post("/api/config")
async def update_config(req: Request):
    data = await req.json()
    current = settings_manager.config
    current.update(data)
    with open("settings.json", "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2, ensure_ascii=False)
    settings_manager.reload()
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    if not os.path.exists("history.json"):
        with open("history.json", "w") as f: json.dump([], f)
    
    logger.info("RAI Server starting on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)