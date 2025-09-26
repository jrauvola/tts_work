## Real-time TTS with Streaming Alignments

This repo implements a TTS microservice that streams audio and per-character alignments over a WebSocket gateway. Goal: p50 TTFA < 600 ms.

### Architecture (Mermaid)
```mermaid
flowchart LR
    client[Client] <-->|WS JSON| gateway[Service 1: API Gateway & Orchestrator]
    gateway -->|REST /normalize| textproc[Service 2: Text Processing]
    gateway -->|gRPC SynthesizeStream| tts[Service 3: TTS (Coqui Native Alignments)]

    subgraph Shared
        prom[(Prometheus)]
        graf[(Grafana)]
        jaeger[(Jaeger)]
    end

    gateway --> prom
    textproc --> prom
    tts --> prom
    prom --> graf
```

### Assumptions & Risks
- Models: Coqui TTS with native alignments. Placeholders emulate timing; swap with actual models for performance.
- Audio: 16 kHz PCM16 frames, 20 ms; Opus optional via schema.
- Security: API key on Gateway; mTLS hooks provided for internal gRPC.
- Determinism: fixed seeds where used; warm-up at start.