#!/usr/bin/env python3
"""Colab CLI OAuth 헬퍼 (헤드리스 서버용).

사용법:
  python scripts/colab_oauth.py url          # 인증 URL 출력 + pending 상태 저장
  python scripts/colab_oauth.py finish CODE  # 인증 코드로 토큰 저장
  python scripts/colab_oauth.py status       # 토큰 존재 여부 확인
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

# colab-cli와 동일한 설정 재사용
from colab_cli.auth import PUBLIC_SCOPES, REMOTE_REDIRECT_URI, TOKEN_CONFIG_PATH

PENDING_PATH = Path.home() / ".config" / "colab-cli" / "pending_oauth.json"


def _load_client_config() -> dict:
    config_path = Path.home() / ".colab-cli-oauth-config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    resource = resources.files("colab_cli").joinpath("oauth_config.json")
    return json.loads(resource.read_text())


def cmd_url() -> int:
    flow = InstalledAppFlow.from_client_config(_load_client_config(), PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    auth_url, state = flow.authorization_url(prompt="consent", token_usage="remote")

    pending = {
        "state": state,
        "code_verifier": flow.code_verifier,
    }
    PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    PENDING_PATH.write_text(json.dumps(pending))

    print(auth_url)
    return 0


def cmd_finish(code: str) -> int:
    if not PENDING_PATH.exists():
        print(
            "pending OAuth 상태가 없습니다. 먼저: python scripts/colab_oauth.py url",
            file=sys.stderr,
        )
        return 1

    pending = json.loads(PENDING_PATH.read_text())
    flow = InstalledAppFlow.from_client_config(_load_client_config(), PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    flow.code_verifier = pending["code_verifier"]
    flow.oauth2session.state = pending["state"]

    flow.fetch_token(code=code.strip())
    creds = flow.credentials

    Path(TOKEN_CONFIG_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(TOKEN_CONFIG_PATH).write_text(creds.to_json())
    PENDING_PATH.unlink(missing_ok=True)

    print(f"인증 완료: {TOKEN_CONFIG_PATH}")
    return 0


def cmd_status() -> int:
    if Path(TOKEN_CONFIG_PATH).exists():
        print("authenticated")
        return 0
    print("not_authenticated")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Colab CLI OAuth helper")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("url", help="Print OAuth URL and save pending flow")
    finish_p = sub.add_parser("finish", help="Complete OAuth with authorization code")
    finish_p.add_argument("code", help="Authorization code from Google")
    sub.add_parser("status", help="Check if token exists")

    args = parser.parse_args()
    if args.command == "url":
        return cmd_url()
    if args.command == "finish":
        return cmd_finish(args.code)
    if args.command == "status":
        return cmd_status()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
