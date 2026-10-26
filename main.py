"""Backend do Mobintec: recebe a foto do front e consulta o Workflow do Roboflow.

Contrato (igual ao esperado por script.js):
  POST /api/analisar  (multipart: image, capturadoEm, requisicaoId)
  -> {"mensagem": str, "capturadoEm": str, "requisicaoId": str}
"""
import asyncio
import base64
import logging
import os
import time
from functools import lru_cache

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

WORKSPACE = "guilhermes-workspace-iuyka"
WORKFLOW_ID = "prototipo-de-disponibilidade-de-assentos-1791998395378"
API_URL = "https://serverless.roboflow.com"

# O front descarta resultados com mais de 15 s desde a captura (VALIDADE_RESULTADO),
# então o backend precisa responder antes disso, contando as tentativas.
PRAZO_TOTAL_S = float(os.getenv("PRAZO_TOTAL_S", "10"))
TENTATIVAS = 2
MAX_BYTES = 4 * 1024 * 1024

# Campos grandes ou desnecessários: o front só usa "mensagem".
CAMPOS_EXCLUIDOS = ["output_image", "predictions", "assentos"]

log = logging.getLogger("mobintec")


class RoboflowError(Exception):
    """Falha ao consultar o Workflow."""


class RoboflowTimeout(RoboflowError):
    """O Workflow não respondeu dentro do prazo."""


class RespostaInvalida(RoboflowError):
    """A resposta do Workflow não tem o formato esperado."""


@lru_cache(maxsize=1)
def _cliente() -> InferenceHTTPClient:
    chave = os.environ.get("ROBOFLOW_API_KEY")
    if not chave:
        raise RoboflowError("Defina a variável de ambiente ROBOFLOW_API_KEY.")
    cliente = InferenceHTTPClient(api_url=API_URL, api_key=chave)
    cliente.configure(InferenceConfiguration(api_key_transport="header"))
    return cliente


def _chamar(jpeg: bytes) -> list:
    """Uma chamada síncrona ao Workflow (roda em thread)."""
    return _cliente().run_workflow(
        workspace_name=WORKSPACE,
        workflow_id=WORKFLOW_ID,
        images={"image": base64.b64encode(jpeg).decode("ascii")},
        excluded_fields=CAMPOS_EXCLUIDOS,
    )


def _extrair_mensagem(resultado: object) -> str:
    if not isinstance(resultado, list) or not resultado or not isinstance(resultado[0], dict):
        raise RespostaInvalida("Resposta do Workflow fora do formato esperado.")
    mensagem = resultado[0].get("mensagem")
    if not isinstance(mensagem, str) or not mensagem.strip():
        raise RespostaInvalida("O Workflow não devolveu o campo 'mensagem'.")
    return mensagem.strip()


async def analisar_assentos(jpeg: bytes) -> str:
    """Envia a imagem ao Workflow e devolve a mensagem em português.

    Faz até TENTATIVAS chamadas, todas dentro de PRAZO_TOTAL_S.
    """
    limite = time.monotonic() + PRAZO_TOTAL_S
    ultimo_erro: Exception | None = None

    for tentativa in range(1, TENTATIVAS + 1):
        restante = limite - time.monotonic()
        if restante <= 0:
            break
        try:
            resultado = await asyncio.wait_for(
                asyncio.to_thread(_chamar, jpeg), timeout=restante
            )
            return _extrair_mensagem(resultado)
        except asyncio.TimeoutError:
            raise RoboflowTimeout("O Workflow excedeu o prazo.") from None
        except RespostaInvalida:
            raise
        except Exception as erro:  # falha de rede ou HTTP: tenta de novo
            ultimo_erro = erro
            log.warning("Tentativa %d falhou: %s", tentativa, type(erro).__name__)
            await asyncio.sleep(max(0.0, min(0.5 * tentativa, limite - time.monotonic())))

    if ultimo_erro is None:
        raise RoboflowTimeout("O Workflow excedeu o prazo.")
    raise RoboflowError("Falha ao consultar o Workflow.") from ultimo_erro


app = FastAPI(title="Mobintec backend")

# Ex.: ALLOWED_ORIGIN=https://seu-usuario.github.io
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("ALLOWED_ORIGIN", "http://localhost:8000")],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/saude")
async def saude() -> dict:
    return {"ok": True}


@app.post("/api/analisar")
async def analisar(
    image: UploadFile = File(...),
    capturadoEm: str = Form(...),
    requisicaoId: str = Form(...),
) -> dict:
    if image.content_type != "image/jpeg":
        raise HTTPException(415, "Envie a imagem em JPEG.")

    jpeg = await image.read(MAX_BYTES + 1)
    if len(jpeg) > MAX_BYTES:
        raise HTTPException(413, "Imagem grande demais.")

    try:
        mensagem = await analisar_assentos(jpeg)
    except RoboflowTimeout:
        raise HTTPException(504, "Tempo esgotado ao analisar a imagem.")
    except RoboflowError as erro:
        log.error("Erro Roboflow: %s", erro)
        raise HTTPException(502, "Falha ao analisar a imagem.")

    # O front rejeita a resposta se estes dois campos não voltarem idênticos.
    return {"mensagem": mensagem, "capturadoEm": capturadoEm, "requisicaoId": requisicaoId}
