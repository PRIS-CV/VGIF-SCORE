from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


"""
这是一个用于测试 Gemini 风格视频请求的小脚本。

整体流程：
1. 读取本地视频文件。
2. 将视频内容编码为 Base64，便于内嵌进 JSON 请求体。
3. 可选地向网关查询当前可用模型列表。
4. 调用 Gemini 兼容的 `generateContent` 接口。
5. 将分析结果写入项目目录中的 JSON 文件。
"""


PROJECT_DIR = Path(__file__).resolve().parent

# 当命令行没有传入参数时，使用下面这些默认值。
DEFAULT_API_KEY = os.environ.get("VGIF_API_KEY", "")
DEFAULT_BASE_URL = os.environ.get("VGIF_BASE_URL", "")
DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_VIDEO_PATH = PROJECT_DIR / "01_commercial-product-showcase_873631287184461890_1.mp4"
DEFAULT_PROMPT = "Please describe this video file."
DEFAULT_TIMEOUT = 180
DEFAULT_RETRIES = 2


def parse_args() -> argparse.Namespace:
    """读取命令行参数，允许你在运行时覆盖默认配置。"""
    parser = argparse.ArgumentParser(
        description="将本地视频发送到 Gemini 兼容代理接口，并把分析结果写入 JSON 文件。"
    )
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="代理接口使用的 API Key。")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="代理接口的基础地址，不要以斜杠结尾。",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="要调用的模型 ID。")
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="本地视频文件路径。",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="随视频一起发送给模型的提示词。")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="单次 HTTP 请求超时时间，单位为秒。",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="主请求失败后，额外重试的次数。",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="跳过 /v1/models 预检查。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="同时在终端打印更多调试信息。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="结果 JSON 文件路径；不传时默认写入项目目录。",
    )
    return parser.parse_args()


def build_session(api_key: str) -> requests.Session:
    """
    创建一个可复用的 HTTP 会话。

    这里将 `trust_env` 设为 `False`，是为了避免 requests 自动读取本机代理
    环境变量，从而把“本地代理超时”误判成“目标接口不可用”。
    """
    if not api_key:
        raise ValueError("Missing API key. Set VGIF_API_KEY or pass --api-key.")
    session = requests.Session()
    session.trust_env = False
    session.headers.update(
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    )
    return session


def guess_mime_type(file_path: Path) -> str:
    """根据文件名推断 MIME 类型；如果失败就使用保底值。"""
    mime_type, _ = mimetypes.guess_type(file_path.name)
    return mime_type or "application/octet-stream"


def encode_file_to_base64(file_path: Path) -> str:
    """读取本地文件，并将其编码为 Base64 字符串。"""
    with file_path.open("rb") as file_obj:
        raw_bytes = file_obj.read()
    return base64.b64encode(raw_bytes).decode("utf-8")


def fetch_available_models(
    session: requests.Session,
    base_url: str,
    timeout: int,
) -> list[str]:
    """获取网关当前可用的模型列表。"""
    response = session.get(f"{base_url}/v1/models", timeout=timeout)
    response.raise_for_status()

    payload = response.json()
    return [item["id"] for item in payload.get("data", []) if "id" in item]


def print_model_check(model: str, models: list[str]) -> None:
    """仅当当前模型不在网关列表中时，给出提醒。"""
    if model in models:
        return

    print(f"警告：当前模型 '{model}' 不在网关可用列表中。", file=sys.stderr)

    similar_models = [name for name in models if "gemini-3" in name or "gemini3" in name]
    if similar_models:
        print(
            "可参考的相近模型："
            + json.dumps(similar_models, ensure_ascii=False),
            file=sys.stderr,
        )


def build_request_payload(
    prompt: str,
    model: str,
    mime_type: str,
    video_base64: str,
) -> dict[str, Any]:
    """
    构造 Gemini 风格的请求体。

    视频通过 `inline_data` 字段传递，也就是把文件内容直接内嵌到 JSON 中。
    """
    return {
        "model": model,
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": video_base64,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0,
        },
    }


def send_generate_content_request(
    session: requests.Session,
    base_url: str,
    model: str,
    payload: dict[str, Any],
    timeout: int,
    retries: int,
) -> requests.Response:
    """
    调用代理网关暴露出来的 Gemini 兼容接口。

    如果遇到超时或临时网络异常，会按照 `retries` 指定的次数自动重试。
    """
    endpoint = f"{base_url}/v1beta/models/{model}:generateContent"

    last_error: requests.RequestException | None = None
    total_attempts = retries + 1

    for attempt in range(1, total_attempts + 1):
        try:
            return session.post(endpoint, json=payload, timeout=timeout)
        except requests.Timeout as exc:
            last_error = exc
            if attempt == total_attempts:
                break
            time.sleep(min(attempt * 2, 5))
        except requests.RequestException as exc:
            last_error = exc
            if attempt == total_attempts:
                break
            time.sleep(1)

    assert last_error is not None
    raise last_error


def parse_json_response(response: requests.Response) -> dict[str, Any] | None:
    """尽量将响应解析为 JSON；如果不是合法 JSON，就返回 None。"""
    try:
        return response.json()
    except ValueError:
        return None


def extract_first_text(response_payload: dict[str, Any]) -> str | None:
    """从 Gemini 响应中提取第一段文本描述。"""
    for candidate in response_payload.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                return text
    return None


def strip_large_debug_fields(response_payload: dict[str, Any]) -> dict[str, Any]:
    """
    在打印调试 JSON 前移除过大的字段，避免终端输出过长。
    """
    cleaned_payload = json.loads(json.dumps(response_payload))

    for candidate in cleaned_payload.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            if "thoughtSignature" in part:
                part["thoughtSignature"] = "<已省略，便于阅读>"

    return cleaned_payload


def print_verbose_request_info(video_path: Path, mime_type: str, model: str) -> None:
    """在详细模式下打印本次请求的关键信息。"""
    print("请求信息：")
    print(f"  视频文件：{video_path}")
    print(f"  MIME 类型：{mime_type}")
    print(f"  模型：{model}")


def build_output_path(video_path: Path, custom_output: Path | None) -> Path:
    """计算结果 JSON 文件的输出路径。"""
    if custom_output is not None:
        return custom_output.resolve()
    return PROJECT_DIR / f"{video_path.stem}_description.json"


def build_result_payload(
    video_path: Path,
    output_path: Path,
    prompt: str,
    model: str,
    response: requests.Response,
    response_payload: dict[str, Any],
    description: str | None,
) -> dict[str, Any]:
    """整理最终写入 JSON 文件的数据结构。"""
    return {
        "success": bool(response.ok and description),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "video_file": str(video_path),
        "output_file": str(output_path),
        "prompt": prompt,
        "model": model,
        "model_version": response_payload.get("modelVersion"),
        "http_status": response.status_code,
        "response_id": response_payload.get("responseId"),
        "description": description,
        "usage_metadata": response_payload.get("usageMetadata"),
    }


def write_result_json(output_path: Path, result_payload: dict[str, Any]) -> None:
    """将结果写入 JSON 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    # 第一步：读取命令行参数，并准备好要处理的视频路径。
    args = parse_args()
    video_path = args.video.resolve()
    output_path = build_output_path(video_path, args.output)

    # 第二步：先确认视频文件确实存在，避免后续编码阶段直接报错。
    if not video_path.exists():
        print(f"错误：找不到视频文件：{video_path}", file=sys.stderr)
        return 1

    # 第三步：推断视频的 MIME 类型，并创建用于发送 HTTP 请求的会话对象。
    mime_type = guess_mime_type(video_path)
    session = build_session(args.api_key)

    # 第四步：可选地检查当前模型是否在网关可用列表中。
    if not args.skip_model_check:
        try:
            models = fetch_available_models(session, args.base_url, args.timeout)
            if args.verbose:
                print("网关可用模型：")
                print(json.dumps(models, ensure_ascii=False, indent=2))
                print()
            print_model_check(args.model, models)
        except requests.RequestException as exc:
            print(f"警告：获取模型列表失败：{exc}", file=sys.stderr)

    # 第五步：在详细模式下，把这次请求的关键信息打印出来，便于调试。
    if args.verbose:
        print_verbose_request_info(video_path, mime_type, args.model)

    # 第六步：读取视频并转成 Base64，供后面的 JSON 请求体使用。
    try:
        video_base64 = encode_file_to_base64(video_path)
    except OSError as exc:
        print(f"错误：读取视频文件失败：{exc}", file=sys.stderr)
        return 1

    # 第七步：将提示词、模型名和视频内容一起组装成请求体。
    request_payload = build_request_payload(
        prompt=args.prompt,
        model=args.model,
        mime_type=mime_type,
        video_base64=video_base64,
    )

    # 第八步：向 Gemini 兼容接口发送请求；如果网络临时不稳定，内部会自动重试。
    try:
        response = send_generate_content_request(
            session=session,
            base_url=args.base_url,
            model=args.model,
            payload=request_payload,
            timeout=args.timeout,
            retries=args.retries,
        )
    except requests.RequestException as exc:
        print(f"错误：请求失败：{exc}", file=sys.stderr)
        return 1

    # 第九步：优先把响应解析成 JSON，这样后续既能提取描述，也能保存结构化结果。
    response_payload = parse_json_response(response)
    if response_payload is None:
        print("错误：接口返回的不是合法 JSON。", file=sys.stderr)
        print(response.text, file=sys.stderr)
        return 1

    # 第十步：从返回结果中提取第一段描述文本。
    description = extract_first_text(response_payload)

    # 第十一步：整理出最终要写入文件的 JSON 内容，并保存到项目目录中。
    result_payload = build_result_payload(
        video_path=video_path,
        output_path=output_path,
        prompt=args.prompt,
        model=args.model,
        response=response,
        response_payload=response_payload,
        description=description,
    )
    write_result_json(output_path, result_payload)

    # 第十二步：在详细模式下打印清理后的完整响应，方便你调试网关返回内容。
    if args.verbose:
        print(f"\nHTTP {response.status_code}")
        print("写入文件：", output_path)
        print("\n完整响应：")
        print(json.dumps(strip_large_debug_fields(response_payload), ensure_ascii=False, indent=2))

    # 第十三步：如果接口没有成功，或者没有提取到描述，就给出明确报错。
    if not response.ok:
        print(f"错误：HTTP {response.status_code}，详细结果已写入：{output_path}", file=sys.stderr)
        return 1

    if not description:
        print(f"错误：未从响应中提取到描述文本，详细结果已写入：{output_path}", file=sys.stderr)
        return 1

    # 第十四步：成功时只提示结果文件路径，方便你直接去项目目录查看 JSON。
    print(f"描述结果已写入：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
