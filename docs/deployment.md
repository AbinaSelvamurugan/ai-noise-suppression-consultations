# Deployment instructions and constraints

## Local installation

See README.md for Linux/macOS/Windows commands. The reference configuration uses CPU-only PyTorch. `uvicorn backend.app.main:app --host 127.0.0.1 --port 8000`, then `cd frontend && npm run dev`; development Vite proxy forwards `/api` HTTP and WebSocket traffic. Browser microphone permissions normally require a secure context (localhost is allowed by browsers).

## Compose

`docker compose up --build` (requires Docker Engine/Compose and registry network). Backend Python 3.11 is a nonroot process, frontend builds static React assets with Node and serves through nginx. Port 5173 exposes local frontend; nginx forwards `/api` to backend and handles WebSocket upgrade. Checkpoints are read-only mounted from `artifacts/checkpoints` and loaded once on backend startup. Container builds require Docker or npm registry access, and dependency pins may need revisiting on different hardware.

## Production considerations

The local Compose file is not an internet-ready production deployment. Place a proper TLS reverse proxy in front; configure explicit allowed CORS/WS origins to the public host; implement authentication, session-level rate limits, authorization and user consent; add audio deletion/review procedures and security monitoring with no raw waveform logging. Limit ingress upload sizes and WebSocket session connections at the reverse proxy as well as application limits; set CPU/memory limits, use process isolation, encrypt backups and restrict checkpoint provisioning. Test real network and audio-device latency; HTTPS/WSS is required away from localhost. Evaluate applicable clinical/privacy regulation with legal professionals.

The backend is one Uvicorn process with threadpool-concurrent request handling and up to `CONSULT_MAX_CONCURRENT` processing operations. GPU use is optional when explicitly configured and a compatible CUDA PyTorch install exists. GPU contention is not benchmarked. A new process reload is required for newly placed trained checkpoints. An application worker does not train the model. Uploaded samples are decoded and processed in memory, not saved; reverse proxies and access logs must not capture full payloads.
