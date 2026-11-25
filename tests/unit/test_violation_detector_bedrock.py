"""bedrock モード (Amazon Bedrock の Converse API で、予備経路と同じ 2 段の判定) のテスト。

Bedrock のクライアントは偽物 (MagicMock) に差し替え、AWS には接続しない。
"""
import json
from unittest.mock import MagicMock

import pytest
import requests

from app_inspect.services.violation_detector import (
    BEDROCK_TEMPERATURE,
    ViolationDetector,
    _DECISION_TEXT_PLACEHOLDER,
)

MODEL_ID = "amazon.nova-lite-v1:0"


@pytest.fixture
def articles_file(tmp_path):
    data = {
        "articles": [
            {"id": "11-v", "article": "第11条(v)", "content": "宗教・マルチ商法等の勧誘行為", "category": "勧誘行為"},
            {"id": "11-xiii", "article": "第11条(xiii)", "content": "外部コミュニティへの勧誘や宣伝広告", "category": "宣伝・勧誘"},
            {"id": "11-xvi", "article": "第11条(xvi)", "content": "その他不適切な行為", "category": "不適切行為"},
            {"id": "reg-3", "article": "共通規程第3条", "content": "アカウントの不正貸与", "category": "不正アクセス"},
        ]
    }
    path = tmp_path / "articles.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


@pytest.fixture
def openai_client():
    client = MagicMock()
    client.api_key = "sk-test"
    return client


def _tool_response(name: str, answer: dict) -> dict:
    """Converse API の応答 (ツールを呼んだ形)"""
    return {
        "output": {"message": {"role": "assistant", "content": [
            {"toolUse": {"toolUseId": "t1", "name": name, "input": answer}},
        ]}},
        "stopReason": "tool_use",
        "usage": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120},
        "metrics": {"latencyMs": 300},
    }


def _detector(articles_file, openai_client, bedrock_client, mode="bedrock"):
    return ViolationDetector(
        openai_client=openai_client,
        judge_model=MODEL_ID,
        articles_path=articles_file,
        detection_mode=mode,
        bedrock_client=bedrock_client,
    )


@pytest.mark.parametrize("model", ["amazon.nova-lite-v1:0", "apac.amazon.nova-lite-v1:0", "us.amazon.nova-micro-v1:0"])
def test_auto_mode_resolves_bedrock_for_nova(monkeypatch, articles_file, openai_client, model):
    monkeypatch.delenv("DETECTION_MODE", raising=False)
    det = ViolationDetector(openai_client=openai_client, judge_model=model, articles_path=articles_file)
    assert det.detection_mode == "bedrock"


def test_explicit_bedrock_mode(monkeypatch, articles_file, openai_client):
    monkeypatch.setenv("DETECTION_MODE", " BEDROCK ")
    det = ViolationDetector(openai_client=openai_client, judge_model=MODEL_ID, articles_path=articles_file)
    assert det.detection_mode == "bedrock"


# warn → 2 段目で条文を選ぶ。OpenAI と Decisions API は呼ばない。ツールを toolChoice で呼ばせ、temperature は 0
def test_bedrock_warn_pipeline(mocker, articles_file, openai_client):
    post = mocker.patch.object(requests, "post")
    bedrock = MagicMock()
    bedrock.converse.side_effect = [
        _tool_response("warn_decision_cands", {
            "decision": "warn", "candidates": ["11-xiii", "11-v", "11-xvi"], "reason": "外部グループへの勧誘"}),
        _tool_response("article_choice", {"article_id": "11-xiii", "reason": "WhatsApp への勧誘"}),
    ]

    result = _detector(articles_file, openai_client, bedrock).detect("Join my WhatsApp group!")

    post.assert_not_called()
    openai_client.chat.completions.create.assert_not_called()
    assert bedrock.converse.call_count == 2
    assert result.is_violation is True
    assert result.method == "Bedrock"
    assert result.article_id == "11-xiii"
    assert result.category == "宣伝・勧誘"
    assert "[bedrock] WhatsApp への勧誘" in result.reason

    first = bedrock.converse.call_args_list[0].kwargs
    assert first["modelId"] == MODEL_ID
    assert first["inferenceConfig"]["temperature"] == BEDROCK_TEMPERATURE == 0.0
    prompt = first["messages"][0]["content"][0]["text"]
    assert "Join my WhatsApp group!" in prompt
    assert _DECISION_TEXT_PLACEHOLDER not in prompt
    tool = first["toolConfig"]["tools"][0]["toolSpec"]
    assert first["toolConfig"]["toolChoice"] == {"tool": {"name": tool["name"]}}
    schema = tool["inputSchema"]["json"]
    # Nova は最上位に type・properties・required しか書けない
    assert set(schema) <= {"type", "properties", "required"}
    assert schema["properties"]["decision"]["enum"] == ["warn", "safe"]

    second = bedrock.converse.call_args_list[1].kwargs
    choice_schema = second["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]
    assert choice_schema["properties"]["article_id"]["enum"] == ["11-xiii", "11-v", "11-xvi"]


def test_bedrock_safe_skips_stage2(articles_file, openai_client):
    bedrock = MagicMock()
    bedrock.converse.return_value = _tool_response("warn_decision_cands", {
        "decision": "safe", "candidates": ["11-xiii", "11-v", "11-xvi"], "reason": "自己紹介"})

    result = _detector(articles_file, openai_client, bedrock).detect("Hello, I'm from Kenya.")

    assert bedrock.converse.call_count == 1
    assert result.is_violation is False
    assert result.article_id is None
    assert result.method == "Bedrock"
    assert result.reason == "[bedrock] 自己紹介"


# warn/safe 以外の値は、黙って safe にせず判定の失敗にする (評価で取りこぼしと分けて数えられるように)
def test_bedrock_invalid_decision_is_error(articles_file, openai_client):
    bedrock = MagicMock()
    bedrock.converse.return_value = _tool_response("warn_decision_cands", {
        "decision": "maybe", "candidates": [], "reason": "?"})

    result = _detector(articles_file, openai_client, bedrock).detect("text")

    assert result.is_violation is False
    assert result.reason.startswith("LLM判定エラー")


def test_bedrock_no_tool_use_is_error(articles_file, openai_client):
    bedrock = MagicMock()
    bedrock.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "I think it is fine."}]}},
        "stopReason": "end_turn",
    }

    result = _detector(articles_file, openai_client, bedrock).detect("text")

    assert result.is_violation is False
    assert result.reason.startswith("LLM判定エラー")
    assert "end_turn" in result.reason


def test_bedrock_client_exception_is_error(articles_file, openai_client):
    bedrock = MagicMock()
    bedrock.converse.side_effect = RuntimeError("ThrottlingException")

    result = _detector(articles_file, openai_client, bedrock).detect("text")

    assert result.is_violation is False
    assert result.reason.startswith("LLM判定エラー")
    assert "ThrottlingException" in result.reason


# 2 段目で候補に無い条文を返したら、候補の先頭を使う (予備経路と同じ扱い)
def test_bedrock_stage2_out_of_candidates_falls_back_to_first(articles_file, openai_client):
    bedrock = MagicMock()
    bedrock.converse.side_effect = [
        _tool_response("warn_decision_cands", {
            "decision": "warn", "candidates": ["11-v", "11-xiii"], "reason": "ネットワークビジネスの勧誘"}),
        _tool_response("article_choice", {"article_id": "not-an-article", "reason": "x"}),
    ]

    result = _detector(articles_file, openai_client, bedrock).detect("Earn money with our MLM!")

    assert result.is_violation is True
    assert result.article_id == "11-v"
    assert "ネットワークビジネスの勧誘" in result.reason


# 候補に条文一覧に無い ID や重複があれば除く。全部無効なら 11-xvi
def test_bedrock_candidate_sanitization(articles_file, openai_client):
    bedrock = MagicMock()
    bedrock.converse.side_effect = [
        _tool_response("warn_decision_cands", {
            "decision": "warn", "candidates": ["bogus", "bogus"], "reason": "不適切"}),
        _tool_response("article_choice", {"article_id": "11-xvi", "reason": "不適切な投稿"}),
    ]

    result = _detector(articles_file, openai_client, bedrock).detect("text")

    choice_schema = bedrock.converse.call_args_list[1].kwargs["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]
    assert choice_schema["properties"]["article_id"]["enum"] == ["11-xvi"]
    assert result.article_id == "11-xvi"


# クライアントを渡さなければ boto3 から作る (BEDROCK_REGION があればその地域)
def test_bedrock_client_created_lazily(mocker, monkeypatch, articles_file, openai_client):
    monkeypatch.setenv("BEDROCK_REGION", "ap-northeast-1")
    fake = MagicMock()
    fake.converse.return_value = _tool_response("warn_decision_cands", {
        "decision": "safe", "candidates": [], "reason": "雑談"})
    make = mocker.patch("boto3.client", return_value=fake)

    det = _detector(articles_file, openai_client, None)
    make.assert_not_called()
    det.detect("text")

    make.assert_called_once()
    assert make.call_args.args[0] == "bedrock-runtime"
    assert make.call_args.kwargs["region_name"] == "ap-northeast-1"
