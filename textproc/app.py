import os
import re
from typing import Dict

from fastapi import FastAPI, HTTPException
from prometheus_client import make_asgi_app
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel

try:
    from num2words import num2words  # type: ignore
except Exception:  # pragma: no cover - optional at runtime, required in container
    def num2words(x: int):
        return str(x)


APP_NAME = os.environ.get("APP_NAME", "textproc")

app = FastAPI(title="Text Processing Service", version="0.1.0")

# Tracing (Jaeger) optional
def _init_tracing() -> None:
    try:
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter  # type: ignore
        provider = TracerProvider()
        jaeger_host = os.environ.get("JAEGER_AGENT_HOST", "jaeger")
        jaeger_port = int(os.environ.get("JAEGER_AGENT_PORT", "6831"))
        exporter = JaegerExporter(agent_host_name=jaeger_host, agent_port=jaeger_port)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
    except Exception:
        pass

_init_tracing()


class NormalizeIn(BaseModel):
    text: str


class NormalizeOut(BaseModel):
    normalized_text: str


ABBREVIATIONS = [
    (re.compile(r"\bMr\.", re.IGNORECASE), "Mister"),
    (re.compile(r"\bMrs\.", re.IGNORECASE), "Misses"),
    (re.compile(r"\bDr\.", re.IGNORECASE), "Doctor"),
    (re.compile(r"\bSt\.", re.IGNORECASE), "Street"),
    (re.compile(r"\bAve\.", re.IGNORECASE), "Avenue"),
]


def expand_abbreviations(text: str) -> str:
    result = text
    for pattern, repl in ABBREVIATIONS:
        result = pattern.sub(repl, result)
    return result


NUM_RE = re.compile(r"(?<!power of )\b(\d{2,9})\b")


def numbers_to_words(text: str) -> str:
    def _replace(m: re.Match[str]) -> str:
        try:
            n = int(m.group(1))
            # American English, avoid 'and' and hyphens for test expectations
            w = num2words(n, to='cardinal', lang='en_US')
            w = w.replace('-', ' ').replace(' and ', ' ')
            return w
        except Exception:
            return m.group(0)

    return NUM_RE.sub(_replace, text)


MATH_INT_RE = re.compile(r"\\+int_\{?(?P<a>[^}^\s]+)\}?\^\{?(?P<b>[^}^\s]+)\}?\s+(?P<body>.+)")


def latex_math_to_speech(text: str) -> str:
    # Minimal heuristic for integrals and exponents
    def exp_to_words(expr: str) -> str:
        return re.sub(r"([A-Za-z])\^(\d+)", lambda m: f"{m.group(1)} to the power of {m.group(2)}", expr)

    def _replace_integral(m: re.Match[str]) -> str:
        a = m.group("a").strip() or "zero"
        b = m.group("b").strip() or "one"
        body = exp_to_words(m.group("body").strip())
        return f"the integral from {a} to {b} of {body}"

    # Integrals first
    text = MATH_INT_RE.sub(_replace_integral, text)
    # Do not globally convert ^N to avoid touching "to 1" phrases
    # Common symbols
    text = text.replace("\\times", " times ").replace("\\cdot", " times ")
    return text


def normalize(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("text must be str")
    text = text.strip()
    text = re.sub(r"[\u0000-\u001F]", " ", text)  # remove control chars
    text = expand_abbreviations(text)
    text = numbers_to_words(text)
    text = latex_math_to_speech(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@app.get("/livez")
async def livez() -> Dict[str, str]:
    return {"status": "ok", "service": APP_NAME}


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ready", "service": APP_NAME}


@app.post("/normalize", response_model=NormalizeOut)
async def normalize_api(body: NormalizeIn) -> NormalizeOut:
    try:
        return NormalizeOut(normalized_text=normalize(body.text))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Prometheus metrics
app.mount("/metrics", make_asgi_app())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("textproc.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8001)), reload=False)


