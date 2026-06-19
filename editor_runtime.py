from __future__ import annotations

import argparse
import socket

import uvicorn

from core.editor.service import create_app


def _resolve_port(host: str, port: int) -> int:
    if port != 0:
        return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenClip editor runtime service")
    parser.add_argument("--host", default="172.18.140.100")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--projects-root", default="processed_videos")
    parser.add_argument("--jobs-dir", default="jobs")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    resolved_port = _resolve_port(args.host, args.port)
    app = create_app(projects_root=args.projects_root, jobs_dir=args.jobs_dir)
    print(f"OpenClip editor service listening on http://{args.host}:{resolved_port}", flush=True)
    uvicorn.run(app, host=args.host, port=resolved_port, log_level=args.log_level)


if __name__ == "__main__":
    main()
