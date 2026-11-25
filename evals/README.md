# evals — 違反検知の評価（promptfoo）

本番の判定のコード（`lambda/app_inspect/services/violation_detector.py` の `ViolationDetector`）を、そのまま [promptfoo](https://promptfoo.dev) から呼んで評価します。
プロンプトだけを評価するのではなく、本番と同じ 2 段の判定（Decisions API → 条文の選択、失敗時の予備経路）を通すので、評価の結果がそのまま本番の判定の精度になります。

- 対象の方式: `decisions`（新方式・本番）、`luna`（予備経路だけ）、`llm_score`（旧方式）、`bedrock`（予備経路と同じ 2 段の判定を Amazon Bedrock で行う。下の「Bedrock で評価する」）
- データ: 発注書 PJT04 の評価用データセット（架空の投稿 354 件。58 のシナリオを 6 言語で書いた対訳と、言語混在 6 件）。**リポジトリには置きません**

---

## 何を計測するか

すべて言語別（ja・en・zh・ar・fr・sw・言語混在）と全体の両方で出します（`summarize.py`）。

| # | 指標 | 定義 |
|---|---|---|
| 1 | **取りこぼし**（最優先） | 警告すべき投稿を「違反なし」とした件数。再現率 = 正しく警告した件数 ÷ 警告すべき投稿の件数 |
| 2 | **誤検知** | 警告しない投稿を「違反」とした件数。適合率 = 正しく警告した件数 ÷ 通知の件数。通知の件数も出す |
| 3 | F1 | 再現率と適合率の調和平均 | 1 つの数で比べるときの参考 |
| 4 | **判定の失敗** | 判定の段が例外・スロットリング・時間切れで結論を出せなかった件数。**取りこぼしと混ぜずに別に数える**。「失敗を取りこぼしに数えた再現率」（最悪の値）も並べる。判定の方法（予備経路に切り替わった件数）と Decisions API の応答コードも出す |
| 5 | 条文の一致 | 正しく警告した投稿のうち、条文が正解と一致した割合 |
| 6 | **言語の間の一致** | 同じシナリオ（6 言語の対訳）で、判定が全言語でそろった割合と、分かれたシナリオ |
| 7 | 判定の安定性 | 同じデータを複数回（`--repeat 3`）判定し、毎回同じ判定になった割合と、回ごとの再現率・適合率 |
| 8 | 判定時間 | 1 投稿の判定時間の中央値・95 パーセンタイル・最大 |
| 9 | 費用 | 1 投稿あたりの平均（トークン数から計算）。警告した投稿と警告しない投稿に分け、**本番の違反の割合（1%・5%）に直した値**と月 3 万投稿の月額も出す |
| 10 | 区分ごとの内訳 | 投稿の種類（`category`）・難しさ（`degree`）・意図（`intent`）・表記（`noise`）・長さ（`length`）ごとの取りこぼし・誤検知・失敗 |

あわせて、間違えた投稿の id ごとに、警告の確率（P(warn)）と条文の候補の上位 3 つを出します。新方式は「違反なし」と判定すると条文を付けない（`article_id` が `null`）ので、取りこぼしの分析には Decisions API の条文の候補を使います。

件数が少ない（言語ごとに 58 件、うち警告すべきは 25 件）ので、**率と件数を必ず並べて**報告してください（言語ごとの再現率は 1 件で 0.04 動きます）。

単価（USD / 100 万トークン）は `docs/research/20261013_detection_model_migration.md` の 6.2 の値です（`promptfoo/harness.py` の `PRICES`）。

---

## 構成

```
evals/
├── README.md               # この文書
├── PLAN_new_eval.md        # 作り直しの計画と引継ぎ
├── package.json            # promptfoo（版を固定）
└── promptfoo/
    ├── promptfooconfig.yml # 評価の設定（プロバイダー・アサーション）
    ├── provider.py         # promptfoo の Python のプロバイダー。ViolationDetector で 1 投稿を判定する
    ├── harness.py          # 判定を呼ぶ共通の部分（記録・再試行・失敗の印・費用）
    ├── load_dataset.py     # データセット（CSV / JSONL）→ promptfoo のテストケース（YAML）
    ├── summarize.py        # promptfoo の結果 → 言語別の指標
    ├── datasets/           # 変換したテストケース（.gitignore 済み。commit しない）
    └── output/             # 結果（投稿の本文を含む。commit しない）
```

---

## 準備

`lecture_slack_inspection/` で。

```bash
# 1. promptfoo（Node.js 18 以上）
cd evals && npm install && cd ..

# 2. Python（判定のコードの依存。uv の例。venv と pip でもよい）
uv venv --python 3.13 evals/.venv
uv pip install --python evals/.venv/bin/python -r lambda/requirements.txt pyyaml

# 3. OpenAI の API キー（環境変数）。本番用のキーは使わない
export OPENAI_API_KEY=sk-...
```

API キーは環境変数 `OPENAI_API_KEY` から読みます（無ければ、リポジトリの `.env` の `LLM_API_KEY`）。キーをファイルに書いて commit しないでください。

### データセット

配布されたデータセット（`eval_posts.csv` または `eval_posts.jsonl`）を手元に置き、テストケースに変換します。条文の ID が `lambda/common/data/articles.json` に無いと止まります。

```bash
cd evals/promptfoo
../.venv/bin/python load_dataset.py <データセットのパス>/eval_posts.jsonl
# → datasets/testcases.yaml（354 件）
```

---

## 実行

```bash
cd evals/promptfoo
export PROMPTFOO_PYTHON=$PWD/../.venv/bin/python   # 判定のコードの依存が入った Python
export PROMPTFOO_DISABLE_TELEMETRY=1

# 1 回だけ
npx promptfoo eval --no-cache -o output/results.json

# 安定性も測る（同じデータを 3 回。費用は 3 倍）
npx promptfoo eval --repeat 3 --no-cache -o output/results.json

# 試しに先頭の数件だけ
npx promptfoo eval --filter-first-n 5 --no-cache -o output/smoke.json
```

- `--no-cache` を付ける（同じ投稿の 2 回目以降も実際に判定させるため）
- 同時に判定する数は `promptfooconfig.yml` の `workers`（既定 4）。スロットリング（429）が多いときは下げる
- 方式を替えるときは `promptfooconfig.yml` の `config.mode`（と `model`）を変える。複数の方式を並べて比べるなら、`providers` に `label` を変えて並べる
- `--share` は使わない（結果に投稿の本文が入るため、外部に送らない）
- 費用の目安: 新方式で 258 件 × 1 回あたり約 0.14 ドル（データセットが 258 件だったときの値。354 件ならその約 1.4 倍）

### 結果の見方

```bash
# 言語別の指標（Markdown）。--csv で言語別の表を CSV にも書く
../.venv/bin/python summarize.py output/results.json --csv output/summary.csv

# 1 件ずつの判定を画面で見る
npx promptfoo view
```

- promptfoo の合否（`Detection`）は「違反かどうかの一致」。判定の失敗は不合格になり、理由に「判定の失敗」と出る。**集計（`summarize.py`）では、失敗を取りこぼしと分けて数える**
- `Article` は参考の指標（合否に入れない）。正しく警告した投稿だけを見る

---

## 判定の失敗と再試行の扱い

- 本番のコードは、判定の段が失敗すると黙って「違反なし」を返す（理由は「LLM判定エラー」で始まる）。評価ではこれを「判定の失敗」とし、`harness.py` が最大 2 回まで判定し直す。それでも失敗したものは `error` として記録する
- Decisions API が一時的なエラー（429・500・502・503・504）を返したときは、予備経路に切り替える前に、同じ経路で最大 3 回再試行する（`Retry-After` があればその秒数、無ければ 3 秒待つ）。評価の結果が API の混雑に左右されないようにするため。503 は OpenAI の資料で「Model temporarily overloaded」（https://developers.openai.com/api/docs/guides/error-codes ）
- **本番のコードは再試行しない**（すぐ予備経路に切り替える）。評価の結果は「API が混雑していないとき」の精度で、本番では混雑の分だけ予備経路での判定が混ざる（課題 ISS-33）

---

## 新しい方式を評価する（例: Amazon Bedrock）

1. `ViolationDetector` に方式を足す（`DETECTION_MODE` の値を増やす）。今の方式は消さない（既存のテストが OpenAI のクライアントを差し替えて動くため）
2. `promptfoo/harness.py` の `MODELS` に方式と既定のモデルを足す。OpenAI 以外のクライアントを使う場合は、`get_client()` と呼び出しの記録（トークン数・時間）、`PRICES` も足す
3. `promptfooconfig.yml` の `providers` に、その方式の行を足す（`label` で区別する）
4. 同じデータセット・同じ指標（`summarize.py`）で、今の方式と比べる

---

## Bedrock で評価する

`bedrock` モードは、予備経路（`luna`）と同じ 2 段の判定（warn/safe と条文の候補 → 条文を 1 つ選ぶ）を、Amazon Bedrock の Converse API で行います。OpenAI の json_schema の代わりに、ツールを 1 つ渡して `toolChoice` で呼ばせ、その入力を答えとして読みます。Bedrock は `enum` を守る保証が無いので、`decision`・条文 ID はコードで確かめ、範囲外なら判定の失敗にします（「違反なし」と混ぜない）。

```bash
cd evals/promptfoo
export PROMPTFOO_PYTHON=$PWD/../.venv/bin/python
export PROMPTFOO_DISABLE_TELEMETRY=1

# AWS の認証情報（講座の AWS 環境のもの）と地域。OpenAI のキーは要らない
export AWS_PROFILE=...            # またはアクセスキーの環境変数
export BEDROCK_REGION=...         # 例: ap-northeast-1

# 費用の計算に使う単価（USD / 100 万トークン）。地域・モデルで違うので、公式の価格ページで確かめて入れる
# 無ければ費用は 0 として数え、警告が出る
export BEDROCK_PRICE_IN=...
export BEDROCK_PRICE_OUT=...

# まず数件で試す → 全件
npx promptfoo eval -c promptfooconfig.bedrock.yml --filter-first-n 5 --no-cache -o output/bedrock_smoke.json
npx promptfoo eval -c promptfooconfig.bedrock.yml --repeat 3 --no-cache -o output/bedrock.json
../.venv/bin/python summarize.py output/bedrock.json --csv output/bedrock_summary.csv
```

- モデルは `promptfooconfig.bedrock.yml` の `model`（無ければ環境変数 `BEDROCK_MODEL_ID`、既定は `amazon.nova-lite-v1:0`）。地域によっては推論プロファイルの ID（`apac.`・`us.` などで始まる）を使う
- `summarize.py` は、Bedrock の呼び出しの結果（`stopReason` か例外の種類）も数える。`tool_use` 以外が多いときは、答えの形が崩れている

---

## 注意

- データセットと結果（`datasets/`・`output/`）は commit しない。結果には投稿の本文が入る
- 評価に使う API キーは開発用のものにする（本番用のキーを使わない）
- このデータセットに合わせて判定方針を調整しすぎると、未知の投稿での精度が下がる（過学習）。データを開発用と評価用に分ける、別の投稿を足して確かめる、などをする
- プロンプトは作業ツリーの版を読む（書き換えた改善がそのまま評価される）。commit 済みの版で評価したいときは、`promptfooconfig.yml` の `config` に `prompts_from_head: true` を書く（このリポジトリでは、作業ツリーの `judge_violation.txt` が壊れていることがある。ISS-22）
