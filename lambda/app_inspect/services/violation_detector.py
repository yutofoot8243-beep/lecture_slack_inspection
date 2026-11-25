"""違反検出モジュール: 全条文 → LLM 判定

旧仕様の RAG (関連条文 top_k=3 抽出) と NG ワード即確定経路は廃止済。
context が十分広いモデルを用いる前提かつ、条文数も少数 (24+α) なので、
articles.json + extra_articles (WS ローカルルール) を全部 LLM に渡す方式に統一。

bedrock モード: 予備経路 (luna) と同じ 2 段の判定 (warn/safe と条文の候補 → 条文を 1 つ選ぶ) を、
Amazon Bedrock の Converse API で行う。OpenAI の json_schema の代わりに、ツールを 1 つだけ渡して
toolChoice で呼ばせ、その入力 (toolUse.input) を JSON の答えとして読む。Bedrock は enum を守る保証が
無いので、答えの値はコードで確かめ、範囲外なら判定の失敗として扱う。
"""
from functools import lru_cache
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DECISIONS_API_URL = "https://api.openai.com/v1/decisions"
DECISIONS_MODEL = "gpt-6-luna"
DECISIONS_TIMEOUT_SECONDS = 10
DECISION_WARN_THRESHOLD = 0.5

TOP_K = 3

# Bedrock (Converse API)。ツール呼び出しでは temperature=0 (greedy) が推奨されている
# (https://docs.aws.amazon.com/nova/latest/userguide/tool-use-definition.html)
BEDROCK_TEMPERATURE = 0.0
BEDROCK_MAX_TOKENS = 512
BEDROCK_CONNECT_TIMEOUT_SECONDS = 5
BEDROCK_READ_TIMEOUT_SECONDS = 20

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_COMMON_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "common", "data")

_PROMPTS_DIR = os.path.join(_DATA_DIR, "prompts")
_RESPONSE_FORMAT_PATH = os.path.join(_PROMPTS_DIR, "judge_violation.response_format.json")

_DECISION_TEXT_PLACEHOLDER = "(input の投稿)"

_RESPONSE_FORMAT_CACHE: Optional[dict] = None

_DEFAULT_JUDGE_PROMPT = """あなたはSlack投稿のガイドライン違反を判定するアシスタントです。

## 関連する規約条文
{{articles_text}}

## 投稿内容
{{text}}

## タスク
この投稿が上記の規約条文に違反しているか判定してください。

## 出力形式（JSON）
{"violation_score": 0から100までの整数, "confidence": 0.0-1.0, "article_id": "該当条文のID", "category": "違反カテゴリ", "reason": "判定理由"}

JSONのみを出力してください。
"""


def _load_json_list(path: str, key: str) -> list:
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f).get(key, [])


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


@lru_cache
def _load_prompt(filename: str, default: str = "") -> str:
    path = os.path.join(_PROMPTS_DIR, filename)
    if os.path.exists(path):
        return _read_text(path)
    return default


def _get_response_format() -> dict:
    """response_format を初回だけロード (無ければ json_object)"""
    global _RESPONSE_FORMAT_CACHE
    if _RESPONSE_FORMAT_CACHE is not None:
        return _RESPONSE_FORMAT_CACHE

    if os.path.exists(_RESPONSE_FORMAT_PATH):
        with open(_RESPONSE_FORMAT_PATH, encoding="utf-8") as f:
            _RESPONSE_FORMAT_CACHE = json.load(f)
    else:
        _RESPONSE_FORMAT_CACHE = {"type": "json_object"}
    return _RESPONSE_FORMAT_CACHE


def _articles_text(articles: list) -> str:
    return "\n".join(
        [f"- {a['id']} {a.get('article', '')}: {a.get('content', '')}" for a in articles]
    )


def _render_template(template: str, **vars: str) -> str:
    """{{ var }} を最小実装で置換 (依存追加なし)"""
    out = template
    for k, v in vars.items():
        out = re.sub(r"{{\s*" + re.escape(k) + r"\s*}}", lambda _m: str(v), out)
    return out


@dataclass
class DetectionResult:
    is_violation: bool
    confidence: float
    method: str
    article_id: Optional[str]
    category: Optional[str]
    reason: str


class ViolationDetector:
    def __init__(
        self,
        openai_client,
        judge_model: str,
        articles_path: Optional[str] = None,
        detection_mode: Optional[str] = None,
        bedrock_client=None,
    ):
        self.client = openai_client
        self.judge_model = judge_model
        # bedrock モードのときだけ使う。渡されなければ初回の呼び出しで boto3 から作る
        self._bedrock_client = bedrock_client

        mode_raw = detection_mode if detection_mode is not None else os.environ.get("DETECTION_MODE")
        mode = mode_raw.strip().lower() if mode_raw else ""
        if not mode:
            mode = "auto"
        elif mode not in ("auto", "llm_score", "decisions", "luna", "bedrock"):
            logger.warning("Unknown detection mode '%s', falling back to 'auto'", mode_raw)
            mode = "auto"

        if mode == "auto":
            judge_model_norm = (self.judge_model or "").strip().lower()
            if judge_model_norm.startswith("gpt-4o-mini"):
                resolved_mode = "llm_score"
            elif "amazon.nova" in judge_model_norm:
                resolved_mode = "bedrock"
            else:
                resolved_mode = "decisions"
            logger.info(
                "detection_mode=%s (auto, judge_model=%s)",
                resolved_mode,
                self.judge_model,
            )
            self.detection_mode = resolved_mode
        else:
            self.detection_mode = mode

        self.articles = _load_json_list(
            articles_path or os.path.join(_COMMON_DATA_DIR, "articles.json"),
            "articles",
        )
        self._article_title_by_id = {a["id"]: a.get("article", a["id"]) for a in self.articles}
        self._article_id_by_title = {
            a.get("article"): a["id"] for a in self.articles if a.get("article")
        }

    def detect(self, text: str, extra_articles: Optional[list] = None) -> DetectionResult:
        """articles.json + extra_articles を全部 LLM に渡して判定。

        extra_articles は WS ローカルルール条文。
        各要素は {id, article, content, [category, regulation]} を持つ想定。
        """
        articles = list(self.articles)
        if extra_articles:
            articles.extend(extra_articles)
            # 追加条文の id ↔ article マッピングを動的に拡張 (表示用解決のため)
            for a in extra_articles:
                aid = a.get("id")
                article = a.get("article", aid)
                if aid:
                    self._article_title_by_id[aid] = article
                if article:
                    self._article_id_by_title[article] = aid

        if self.detection_mode == "decisions":
            try:
                result = self._judge_by_decisions(text, articles)
                method = "Decisions"
            except Exception as e:
                logger.warning("Decisions API failed, falling back to chat completions: %s", e)
                result = self._judge_by_chat_pipeline(text, articles)
                method = "LLM-fallback"
        elif self.detection_mode == "luna":
            result = self._judge_by_chat_pipeline(text, articles)
            method = "Luna"
        elif self.detection_mode == "bedrock":
            result = self._judge_by_chat_pipeline(text, articles)
            method = "Bedrock"
        else:
            result = self._judge_by_llm(text, articles)
            method = "LLM"

        aid = self._normalize_article_id(result.get("article_id"))
        title = self._article_title_by_id.get(aid) if aid else None
        reason = result.get("reason", "")
        if aid and title:
            reason = f"[{aid} {title}] {reason}"

        category = result.get("category")
        if aid:
            for a in articles:
                if a.get("id") == aid:
                    category = a.get("category")
                    break

        return DetectionResult(
            is_violation=bool(result.get("is_violation", False)),
            confidence=float(result.get("confidence", 0.0)),
            method=method,
            article_id=aid,
            category=category,
            reason=reason,
        )

    def _normalize_article_id(self, article_id: Optional[str]) -> Optional[str]:
        if not article_id:
            return None
        s = str(article_id).strip()
        if not s:
            return None

        # 条文一覧にある ID であればそのまま返す (切り詰めない)
        if s in self._article_title_by_id:
            return s

        # よくある「A-001 ...」形式から ID 部分だけ抜く (枝番付き・ローマ数字も対応)
        m = re.match(r"^([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)", s)
        if m:
            candidate = m.group(1)
            if candidate in self._article_title_by_id:
                s = candidate

        # title を返してきた場合は ID に戻す
        if s not in self._article_title_by_id and s in self._article_id_by_title:
            s = self._article_id_by_title[s]

        return s

    def _judge_by_llm(self, text: str, articles: list) -> dict:
        articles_text = _articles_text(articles)

        template = _load_prompt("judge_violation.txt", _DEFAULT_JUDGE_PROMPT)
        prompt = _render_template(template, text=text, articles_text=articles_text)

        messages = [{"role": "user", "content": prompt}]
        response_format = _get_response_format()

        try:
            resp = self.client.chat.completions.create(
                model=self.judge_model,
                messages=messages,
                response_format=response_format,
            )

            content = (resp.choices[0].message.content or "").strip()
            r = json.loads(content) if content else {}

            score = r.get("violation_score", 0)

            return {
                "is_violation": score >= 51,
                "confidence": r.get("confidence", 0.5),
                "article_id": r.get("article_id"),
                "category": r.get("category"),
                "reason": r.get("reason", ""),
            }
        except Exception as e:
            return {
                "is_violation": False,
                "confidence": 0.0,
                "article_id": None,
                "category": None,
                "reason": f"LLM判定エラー: {e}",
            }

    def _choose_article(self, text: str, candidates: list[dict]) -> dict:
        if not candidates:
            raise ValueError("candidates cannot be empty")
        candidates_text = "\n".join(
            [
                f"- {c.get('id', '')} {c.get('article') or ''} [{c.get('category') or ''}]: {c.get('content') or ''}"
                for c in candidates
            ]
        )
        guide = _load_prompt("article_guide.txt")
        template = _load_prompt("judge_article.txt")
        prompt_template = _render_template(
            template, article_guide=guide, candidates_text=candidates_text
        )
        prompt = _render_template(prompt_template, text=text)

        candidate_ids = [c["id"] for c in candidates if "id" in c]
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "article_choice",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "article_id": {"type": "string", "enum": candidate_ids},
                        "reason": {"type": "string"},
                    },
                    "required": ["article_id", "reason"],
                    "additionalProperties": False,
                },
            },
        }

        r = self._structured_json(prompt, response_format)
        if self.detection_mode == "bedrock" and r.get("article_id") not in candidate_ids:
            # Bedrock は enum を守る保証が無い。候補外なら失敗とし、_pick_article の予備 (候補の先頭) に任せる
            raise ValueError(f"article_id is not in candidates: {r.get('article_id')!r}")
        return {
            "article_id": r.get("article_id"),
            "reason": r.get("reason", ""),
        }

    def _structured_json(self, prompt: str, response_format: dict) -> dict:
        """JSON の形を決めて LLM に答えさせ、dict で返す。

        OpenAI: response_format (json_schema) をそのまま渡す (従来どおり)。
        Bedrock: 同じスキーマのツールを 1 つ渡し、toolChoice でそのツールを呼ばせて、入力を答えとして読む。
        """
        if self.detection_mode == "bedrock":
            js = response_format["json_schema"]
            return self._converse_json(prompt, js["name"], js["schema"])

        resp = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_format,
        )
        content = (resp.choices[0].message.content or "").strip()
        return json.loads(content) if content else {}

    def _get_bedrock_client(self):
        if self._bedrock_client is None:
            import boto3
            from botocore.config import Config

            self._bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=os.environ.get("BEDROCK_REGION") or None,
                config=Config(
                    connect_timeout=BEDROCK_CONNECT_TIMEOUT_SECONDS,
                    read_timeout=BEDROCK_READ_TIMEOUT_SECONDS,
                ),
            )
        return self._bedrock_client

    def _converse_json(self, prompt: str, tool_name: str, schema: dict) -> dict:
        # Nova のツールの inputSchema は、最上位に type・properties・required しか書けない
        # (https://docs.aws.amazon.com/nova/latest/userguide/tool-use-definition.html)
        tool_schema = {k: v for k, v in schema.items() if k in ("type", "properties", "required")}
        resp = self._get_bedrock_client().converse(
            modelId=self.judge_model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": BEDROCK_TEMPERATURE, "maxTokens": BEDROCK_MAX_TOKENS},
            toolConfig={
                "tools": [
                    {
                        "toolSpec": {
                            "name": tool_name,
                            "description": "判定の結果を、決まった形で返す",
                            "inputSchema": {"json": tool_schema},
                        }
                    }
                ],
                "toolChoice": {"tool": {"name": tool_name}},
            },
        )
        for block in resp.get("output", {}).get("message", {}).get("content", []):
            tool_use = block.get("toolUse")
            if tool_use and tool_use.get("name") == tool_name:
                answer = tool_use.get("input")
                if isinstance(answer, dict):
                    return answer
        raise ValueError(f"Bedrock response has no toolUse for {tool_name} (stopReason={resp.get('stopReason')})")

    def _pick_article(
        self,
        text: str,
        candidate_ids: list[str],
        articles: list,
        fallback_reason: str,
    ) -> tuple[Optional[str], str]:
        article_by_id = {a["id"]: a for a in articles}
        candidate_dicts = [article_by_id[cid] for cid in candidate_ids if cid in article_by_id]
        try:
            chosen = self._choose_article(text, candidate_dicts)
            return chosen.get("article_id"), chosen.get("reason", "")
        except Exception as e:
            logger.warning("Article choice failed: %s", e)
            fallback_id = candidate_dicts[0]["id"] if candidate_dicts else None
            return fallback_id, fallback_reason

    def _judge_by_decisions(self, text: str, articles: list) -> dict:
        articles_text = _articles_text(articles)

        template = _load_prompt("judge_decision.txt")
        rendered_template = _render_template(template, articles_text=articles_text)

        guide = _load_prompt("article_guide.txt")
        article_instructions = (
            "この投稿が警告の対象だとしたら、最も当てはまる条文を 1 つ選ぶ。\n\n"
            + guide
            + "\n## 条文\n"
            + articles_text
        )

        payload = {
            "model": DECISIONS_MODEL,
            "input": f"## 投稿内容\n```\n{text}\n```",
            "questions": [
                {
                    "type": "choice",
                    "name": "verdict",
                    "instructions": f"{rendered_template}\n運営がこの投稿者に警告を送るべきなら warn、そうでなければ safe を選ぶ。",
                    "choices": [{"value": "warn"}, {"value": "safe"}],
                },
                {
                    "type": "choice",
                    "name": "article",
                    "instructions": article_instructions,
                    "choices": [{"value": a["id"]} for a in articles],
                },
            ],
        }

        resp = requests.post(
            DECISIONS_API_URL,
            headers={
                "Authorization": f"Bearer {self.client.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=DECISIONS_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()

        data = resp.json()
        answers = data.get("answers", [])

        verdict_ans = next((ans for ans in answers if ans.get("name") == "verdict"), None)
        if not verdict_ans:
            raise ValueError("verdict answer is missing in Decisions API response")
        if verdict_ans.get("type") != "choice":
            raise ValueError(f"verdict answer type is not 'choice': {verdict_ans.get('type')}")

        p_warn = None
        for p in verdict_ans.get("probabilities", []):
            if p.get("value") == "warn":
                p_warn = p.get("probability")
                break

        if p_warn is None:
            raise ValueError("warn probability is missing in verdict answer")

        p_warn = float(p_warn)

        article_ans = next((ans for ans in answers if ans.get("name") == "article"), None)
        if not article_ans:
            raise ValueError("article answer is missing in Decisions API response")
        if article_ans.get("type") != "choice":
            raise ValueError(f"article answer type is not 'choice': {article_ans.get('type')}")

        article_probs = article_ans.get("probabilities", [])
        sorted_probs = sorted(
            article_probs,
            key=lambda x: x.get("probability", 0.0),
            reverse=True,
        )
        candidate_ids = [p["value"] for p in sorted_probs if "value" in p][:TOP_K]

        article_by_id = {a["id"]: a for a in articles}
        candidate_ids = [cid for cid in candidate_ids if cid in article_by_id]

        is_warn = (p_warn >= DECISION_WARN_THRESHOLD)
        fallback_reason = f"Decisions API: P(warn)={p_warn:.2f}"

        if is_warn:
            article_id, reason = self._pick_article(
                text, candidate_ids, articles, fallback_reason
            )
        else:
            article_id = None
            reason = fallback_reason

        return {
            "is_violation": is_warn,
            "confidence": p_warn,
            "article_id": article_id,
            "reason": reason,
            "candidates": candidate_ids,
        }

    def _judge_by_chat_pipeline(self, text: str, articles: list) -> dict:
        articles_text = _articles_text(articles)

        template = _load_prompt("judge_decision.txt")
        part1 = _render_template(template, articles_text=articles_text).replace(
            _DECISION_TEXT_PLACEHOLDER, text
        )
        part2 = _load_prompt("article_guide.txt")
        part3 = _load_prompt("judge_decision_chat_output.txt")
        prompt = part1 + part2 + part3

        article_ids = [a["id"] for a in articles]
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "warn_decision_cands",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "decision": {"type": "string", "enum": ["warn", "safe"]},
                        "candidates": {
                            "type": "array",
                            "items": {"type": "string", "enum": article_ids},
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["decision", "candidates", "reason"],
                    "additionalProperties": False,
                },
            },
        }

        tag = "[bedrock]" if self.detection_mode == "bedrock" else "[luna]"
        try:
            r = self._structured_json(prompt, response_format)

            decision = r.get("decision")
            if self.detection_mode == "bedrock" and decision not in ("warn", "safe"):
                # enum の外の値を黙って safe にしない (取りこぼしと区別できなくなるため)
                raise ValueError(f"decision is not warn/safe: {decision!r}")
            is_warn = (decision == "warn")
            step1_reason = r.get("reason", "")

            valid_ids = {a["id"] for a in articles}
            raw_candidates = r.get("candidates", [])
            if not isinstance(raw_candidates, list):
                raw_candidates = []

            candidate_ids = []
            for cid in raw_candidates:
                if cid in valid_ids and cid not in candidate_ids:
                    candidate_ids.append(cid)
                    if len(candidate_ids) == TOP_K:
                        break

            if not candidate_ids:
                if "11-xvi" in valid_ids:
                    candidate_ids = ["11-xvi"]
                elif articles:
                    candidate_ids = [articles[0]["id"]]
                else:
                    candidate_ids = []

            if is_warn:
                article_id, step2_reason = self._pick_article(
                    text, candidate_ids, articles, step1_reason
                )
                final_reason = f"{tag} {step2_reason}"
            else:
                article_id = None
                final_reason = f"{tag} {step1_reason}"

            return {
                "is_violation": is_warn,
                "confidence": 0.8 if is_warn else 0.2,
                "article_id": article_id,
                "reason": final_reason,
                "candidates": candidate_ids,
            }
        except Exception as e:
            return {
                "is_violation": False,
                "confidence": 0.0,
                "article_id": None,
                "category": None,
                "reason": f"LLM判定エラー: {e}",
            }
