"""Backend MOBINTEC — demonstração de câmera do celular e câmera fixa.

Celular:
    POST /api/analisar

Ponte da ESP32-CAM:
    POST /api/cameras/onibus-demo/captura

Consulta pelo site:
    GET /api/cameras/onibus-demo/resultado

A última leitura da câmera fixa fica somente na memória.
Use uma instância e um processo para esta demonstração.
"""

import asyncio
import base64
import io
import logging
import os
import secrets
import time

from datetime import datetime, timezone
from functools import lru_cache

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from inference_sdk import (
    InferenceConfiguration,
    InferenceHTTPClient,
)


# ---------- Configuração ----------

WORKSPACE = "guilhermes-workspace-iuyka"

WORKFLOW_ID = (
    "prototipo-de-disponibilidade-de-assentos-1791998395378"
)

API_URL = "https://serverless.roboflow.com"

CAMERA_FIXA_ID = "onibus-demo"

VALIDADE_S = 15
MAX_BYTES = 4 * 1024 * 1024
MAX_PIXELS = 4_000_000

PRAZO_TOTAL_S = float(
    os.getenv("PRAZO_TOTAL_S", "12")
)

CELULAR_PUBLICO = (
    os.getenv("CELULAR_PUBLICO", "false").lower() == "true"
)

MAX_ANALISES_DIA = int(
    os.getenv("MAX_ANALISES_DIA", "100")
)

INTERVALO_MINIMO_S = 5

ORIGENS = [
    origem.strip()
    for origem in os.getenv(
        "ALLOWED_ORIGIN",
        "https://pedraoalvess.github.io",
    ).split(",")
    if origem.strip()
]

log = logging.getLogger("mobintec")
log_diagnostico = logging.getLogger("uvicorn.error")


# ---------- Aplicação ----------

app = FastAPI(title="MOBINTEC — câmeras")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGENS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# Estado somente desta instância.
ultima_fixa = None
ultima_fixa_ts = 0.0

ultima_inicio = 0.0
dia_uso = ""
quantidade_uso = 0
tarefa_inferencia = None


# ---------- Roboflow ----------

@lru_cache(maxsize=1)
def cliente():
    chave = os.environ.get("ROBOFLOW_API_KEY")

    if not chave:
        raise RuntimeError(
            "ROBOFLOW_API_KEY não configurada."
        )

    return InferenceHTTPClient(
        api_url=API_URL,
        api_key=chave,
    ).configure(
        InferenceConfiguration(
            api_key_transport="header"
        )
    )


def chamar_workflow(jpeg):
    inicio = time.monotonic()
    status = "erro"

    try:
        resultado = cliente().run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={
                "image": base64.b64encode(jpeg).decode("ascii")
            },
            excluded_fields=[
                "output_image",
                "predictions",
                "assentos",
            ],
        )
        status = "ok"
        return resultado

    finally:
        log_diagnostico.info(
            "diagnostico etapa=preparacao_local_e_chamada_run_workflow "
            "duracao_ms=%.0f status=%s",
            (time.monotonic() - inicio) * 1000,
            status,
        )


def mensagem_segura(resultado):
    if (
        not isinstance(resultado, list)
        or not resultado
        or not isinstance(resultado[0], dict)
    ):
        raise RuntimeError(
            "Formato inesperado do Workflow."
        )

    saida = resultado[0]

    if not isinstance(saida.get("mensagem"), str):
        raise RuntimeError(
            "Workflow sem mensagem."
        )

    selecionado = saida.get("assento_selecionado")

    if not isinstance(selecionado, dict):
        raise RuntimeError(
            "Workflow sem assento_selecionado válido."
        )

    if not selecionado:
        return (
            "Não identifiquei um candidato a assento livre "
            "nesta imagem. Isso não significa que todos "
            "estejam ocupados."
        )

    if selecionado.get("estado") != "livre":
        raise RuntimeError(
            "Estado de candidato inesperado."
        )

    lado = selecionado.get("lado_na_imagem")

    lados_permitidos = {
        "à esquerda da imagem",
        "à direita da imagem",
        "na região central da imagem",
    }

    if lado not in lados_permitidos:
        raise RuntimeError(
            "Localização visual inesperada."
        )

    # A posição da caixa não mede distância física.
    return (
        "Identifiquei um candidato a assento livre "
        + lado
        + ". A disponibilidade não foi confirmada; "
        "o modelo pode confundir encostos com lugares livres. "
        "Confirme com ajuda antes de sentar."
    )


def consumir_excecao(tarefa, inicio):
    # Registra somente o desfecho da tarefa, sem inferir a ordem do 504.
    if tarefa.cancelled():
        status = "cancelada"
        tipo_erro = "nenhum"
    else:
        erro = tarefa.exception()
        status = "erro" if erro else "ok"
        tipo_erro = type(erro).__name__ if erro else "nenhum"

    log_diagnostico.info(
        "diagnostico etapa=tarefa_inferencia_concluida "
        "duracao_ms=%.0f status=%s tipo_erro=%s",
        (time.monotonic() - inicio) * 1000,
        status,
        tipo_erro,
    )


async def analisar_imagem(jpeg):
    global tarefa_inferencia
    global ultima_inicio
    global dia_uso
    global quantidade_uso

    if tarefa_inferencia and not tarefa_inferencia.done():
        raise HTTPException(
            status_code=429,
            detail=(
                "Uma análise ainda está em andamento. Aguarde."
            ),
        )

    agora = time.monotonic()

    if agora - ultima_inicio < INTERVALO_MINIMO_S:
        raise HTTPException(
            status_code=429,
            detail=(
                "Aguarde alguns segundos antes de outra análise."
            ),
        )

    hoje = datetime.now(timezone.utc).date().isoformat()

    if hoje != dia_uso:
        dia_uso = hoje
        quantidade_uso = 0

    if quantidade_uso >= MAX_ANALISES_DIA:
        raise HTTPException(
            status_code=429,
            detail=(
                "O limite de análises desta demonstração "
                "foi atingido."
            ),
        )

    ultima_inicio = agora
    quantidade_uso += 1

    inicio_tarefa = time.monotonic()
    tarefa = asyncio.create_task(
        asyncio.to_thread(chamar_workflow, jpeg)
    )

    tarefa_inferencia = tarefa
    tarefa.add_done_callback(
        lambda concluida: consumir_excecao(
            concluida,
            inicio_tarefa,
        )
    )

    try:
        # O timeout não cancela a execução remota.
        resultado = await asyncio.wait_for(
            asyncio.shield(tarefa),
            timeout=PRAZO_TOTAL_S,
        )

        inicio_tratamento = time.monotonic()

        try:
            mensagem = mensagem_segura(resultado)
        except Exception as erro:
            log_diagnostico.warning(
                "diagnostico etapa=tratamento_resposta "
                "duracao_ms=%.0f status=erro tipo_erro=%s",
                (time.monotonic() - inicio_tratamento) * 1000,
                type(erro).__name__,
            )
            raise

        log_diagnostico.info(
            "diagnostico etapa=tratamento_resposta "
            "duracao_ms=%.0f status=ok",
            (time.monotonic() - inicio_tratamento) * 1000,
        )
        return mensagem

    except asyncio.TimeoutError:
        log_diagnostico.warning(
            "diagnostico etapa=timeout duracao_ms=%.0f "
            "tarefa_pendente=%s",
            (time.monotonic() - inicio_tarefa) * 1000,
            not tarefa.done(),
        )
        raise HTTPException(
            status_code=504,
            detail=(
                "A análise demorou demais. "
                "Aguarde antes de tentar novamente."
            ),
        )

    except HTTPException:
        raise

    except Exception as erro:
        log.error(
            "Falha na análise: %s",
            type(erro).__name__,
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Não consegui obter uma análise válida "
                "do Roboflow."
            ),
        )


# ---------- Validação ----------

def horario_captura(valor):
    if len(valor) > 80:
        raise HTTPException(
            status_code=400,
            detail="Horário inválido.",
        )

    try:
        data = datetime.fromisoformat(
            valor.replace("Z", "+00:00")
        )

        if data.tzinfo is None:
            raise ValueError()

        timestamp = data.timestamp()

    except (ValueError, OverflowError):
        raise HTTPException(
            status_code=400,
            detail=(
                "Envie capturadoEm com data, hora "
                "e fuso horário."
            ),
        )

    idade = time.time() - timestamp

    if idade < -5:
        raise HTTPException(
            status_code=400,
            detail=(
                "O relógio do aparelho está adiantado. "
                "Ative data e hora automáticas."
            ),
        )

    if idade >= VALIDADE_S:
        raise HTTPException(
            status_code=409,
            detail=(
                "A captura já expirou. "
                "Capture uma imagem nova."
            ),
        )

    return timestamp


async def ler_jpeg(image):
    try:
        if image.content_type != "image/jpeg":
            raise HTTPException(
                status_code=415,
                detail="Envie uma imagem JPEG.",
            )

        jpeg = await image.read(MAX_BYTES + 1)

    finally:
        await image.close()

    if not jpeg or len(jpeg) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Imagem vazia ou maior que 4 MB.",
        )

    try:
        with Image.open(io.BytesIO(jpeg)) as foto:
            if foto.format != "JPEG":
                raise ValueError()

            largura, altura = foto.size

            if largura <= 0 or altura <= 0:
                raise ValueError()

            if largura * altura > MAX_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Reduza a resolução da imagem.",
                )

            foto.verify()

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ):
        raise HTTPException(
            status_code=400,
            detail="O arquivo não é um JPEG válido.",
        )

    return jpeg


def montar_resultado(
    mensagem,
    captura,
    timestamp,
    requisicao_id,
):
    restante = max(
        0.0,
        VALIDADE_S - max(
            0.0,
            time.time() - timestamp,
        ),
    )

    recente = restante > 0

    return {
        "estado": "recente" if recente else "expirado",
        "mensagem": mensagem if recente else (
            "A leitura expirou. "
            "Não há informação atual confirmada."
        ),
        "capturadoEm": captura,
        "requisicaoId": requisicao_id,
        "validaPorSegundos": round(restante, 2),
        "disponibilidadeConfirmada": False,
        "disponibilidadePcd": None,
    }


# ---------- Rotas ----------

@app.get("/saude")
async def saude():
    return {
        "ok": True,
        "cameraCelularHabilitada": CELULAR_PUBLICO,
        "cameraFixa": CAMERA_FIXA_ID,
    }


@app.post("/api/analisar")
async def analisar(
    image: UploadFile = File(...),
    capturadoEm: str = Form(...),
    requisicaoId: str = Form(..., max_length=100),
):
    if not CELULAR_PUBLICO:
        raise HTTPException(
            status_code=403,
            detail=(
                "A análise pública pelo celular ainda "
                "não foi habilitada no backend."
            ),
        )

    timestamp = horario_captura(capturadoEm)
    jpeg = await ler_jpeg(image)

    # Revalida depois da leitura do arquivo.
    horario_captura(capturadoEm)

    mensagem = await analisar_imagem(jpeg)

    return montar_resultado(
        mensagem,
        capturadoEm,
        timestamp,
        requisicaoId,
    )


@app.post("/api/cameras/{camera_id}/captura")
async def publicar_fixa(
    camera_id: str,
    image: UploadFile = File(...),
    capturadoEm: str = Form(...),
    requisicaoId: str = Form(..., max_length=100),
    authorization: str | None = Header(default=None),
):
    global ultima_fixa
    global ultima_fixa_ts

    token = os.environ.get("CAMERA_BRIDGE_TOKEN", "")

    if not token:
        raise HTTPException(
            status_code=503,
            detail=(
                "A ponte da câmera ainda não foi configurada."
            ),
        )

    esperado = ("Bearer " + token).encode("utf-8")
    recebido = (authorization or "").encode("utf-8")

    if not secrets.compare_digest(recebido, esperado):
        raise HTTPException(
            status_code=401,
            detail="Ponte não autorizada.",
        )

    if camera_id != CAMERA_FIXA_ID:
        raise HTTPException(
            status_code=404,
            detail="Câmera não cadastrada.",
        )

    timestamp = horario_captura(capturadoEm)
    jpeg = await ler_jpeg(image)

    horario_captura(capturadoEm)

    mensagem = await analisar_imagem(jpeg)

    resultado = montar_resultado(
        mensagem,
        capturadoEm,
        timestamp,
        requisicaoId,
    )

    if (
        resultado["estado"] == "recente"
        and timestamp > ultima_fixa_ts
    ):
        ultima_fixa_ts = timestamp

        ultima_fixa = {
            "mensagem": mensagem,
            "capturadoEm": capturadoEm,
            "timestamp": timestamp,
            "requisicaoId": requisicaoId,
        }

    return resultado


@app.get("/api/cameras/{camera_id}/resultado")
async def resultado_fixa(camera_id: str):
    from fastapi.responses import JSONResponse

    if camera_id != CAMERA_FIXA_ID:
        raise HTTPException(
            status_code=404,
            detail="Câmera não cadastrada.",
        )

    if ultima_fixa is None:
        resultado = {
            "estado": "sem_leitura",
            "mensagem": (
                "A câmera ainda não enviou uma leitura válida."
            ),
            "validaPorSegundos": 0,
        }

    else:
        resultado = montar_resultado(
            ultima_fixa["mensagem"],
            ultima_fixa["capturadoEm"],
            ultima_fixa["timestamp"],
            ultima_fixa["requisicaoId"],
        )

    return JSONResponse(
        content=resultado,
        headers={"Cache-Control": "no-store"},
    )
