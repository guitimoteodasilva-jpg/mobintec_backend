import asyncio
import base64
import logging
import sys
import threading
import types

import pytest
from fastapi import HTTPException


# SDK fictício: importar main não precisa de credenciais nem de rede.
inference_sdk = types.ModuleType("inference_sdk")
inference_sdk.InferenceConfiguration = type(
    "InferenceConfiguration", (), {}
)
inference_sdk.InferenceHTTPClient = type(
    "InferenceHTTPClient",
    (),
    {"__init__": lambda self, **kwargs: None},
)
sys.modules.setdefault("inference_sdk", inference_sdk)

import main


MOCK_RESULT = [
    {
        "mensagem": "resultado fictício para teste",
        "assento_selecionado": {
            "estado": "livre",
            "lado_na_imagem": "à esquerda da imagem",
        },
    }
]


class FakeClient:
    def __init__(self, result=MOCK_RESULT, started=None, release=None):
        self.result = result
        self.started = started
        self.release = release
        self.kwargs = None

    def run_workflow(self, **kwargs):
        self.kwargs = kwargs
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            if not self.release.wait(timeout=3):
                raise TimeoutError("mock wait expired")
        return self.result


@pytest.fixture(autouse=True)
def capture_uvicorn_error(caplog):
    logger = logging.getLogger("uvicorn.error")
    handler = caplog.handler
    already_attached = handler in logger.handlers

    if not already_attached:
        logger.addHandler(handler)

    try:
        yield
    finally:
        if not already_attached:
            logger.removeHandler(handler)


@pytest.fixture(autouse=True)
def reset_analysis_state(monkeypatch):
    monkeypatch.setattr(main, "tarefa_inferencia", None)
    monkeypatch.setattr(main, "ultima_inicio", 0.0)
    monkeypatch.setattr(main, "dia_uso", "")
    monkeypatch.setattr(main, "quantidade_uso", 0)
    monkeypatch.setattr(main, "PRAZO_TOTAL_S", 0.2)


def diagnostic_messages(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "uvicorn.error"
    ]


def test_quick_mock_response_and_durations(monkeypatch, caplog):
    monkeypatch.setattr(main, "cliente", lambda: FakeClient())
    caplog.set_level(logging.INFO, logger="uvicorn.error")

    async def run():
        result = await main.analisar_imagem(b"fake-image")
        await asyncio.sleep(0)
        return result

    result = asyncio.run(run())

    assert "candidato a assento livre" in result
    messages = diagnostic_messages(caplog)
    assert any(
        "etapa=preparacao_local_e_chamada_run_workflow" in msg
        and "status=ok" in msg
        for msg in messages
    )
    assert any(
        "etapa=tratamento_resposta" in msg and "status=ok" in msg
        for msg in messages
    )


def test_base64_and_run_workflow_diagnostics_use_fake_client(
    monkeypatch, caplog
):
    fake_client = FakeClient()
    monkeypatch.setattr(main, "cliente", lambda: fake_client)
    caplog.set_level(logging.INFO, logger="uvicorn.error")

    result = main.chamar_workflow(b"fake-image")

    assert result == MOCK_RESULT
    assert fake_client.kwargs["images"] == {
        "image": base64.b64encode(b"fake-image").decode("ascii")
    }
    messages = diagnostic_messages(caplog)
    assert any(
        "etapa=codificacao_base64" in msg and "status=ok" in msg
        for msg in messages
    )
    assert any(
        "etapa=run_workflow" in msg and "status=ok" in msg
        for msg in messages
    )


def run_blocked_mock(caplog, monkeypatch):
    class BlockingGate:
        def __init__(self):
            self.entered = threading.Event()
            self.released = threading.Event()

        def wait(self, timeout=None):
            self.entered.set()
            return self.released.wait(timeout)

        def set(self):
            self.released.set()

    started = threading.Event()
    gate = BlockingGate()
    monkeypatch.setattr(
        main,
        "cliente",
        lambda: FakeClient(started=started, release=gate),
    )
    monkeypatch.setattr(main, "PRAZO_TOTAL_S", 0.2)
    caplog.set_level(logging.INFO, logger="uvicorn.error")

    wait_for_original = asyncio.wait_for

    async def wait_for_after_mock_blocks(awaitable, timeout):
        assert await asyncio.to_thread(gate.entered.wait, 1)
        assert started.is_set()
        assert not gate.released.is_set()
        return await wait_for_original(awaitable, timeout=timeout)

    monkeypatch.setattr(main.asyncio, "wait_for", wait_for_after_mock_blocks)

    async def run():
        with pytest.raises(HTTPException) as caught:
            await main.analisar_imagem(b"fake-image")

        assert caught.value.status_code == 504
        assert not main.tarefa_inferencia.done()

        gate.set()
        await main.tarefa_inferencia
        await asyncio.sleep(0)

    try:
        asyncio.run(run())
    finally:
        gate.set()


def test_timeout_records_pending_task(monkeypatch, caplog):
    run_blocked_mock(caplog, monkeypatch)

    messages = diagnostic_messages(caplog)
    assert any(
        "etapa=timeout" in msg and "tarefa_pendente=True" in msg
        for msg in messages
    )


def test_completion_after_timeout_is_logged(monkeypatch, caplog):
    run_blocked_mock(caplog, monkeypatch)

    messages = diagnostic_messages(caplog)
    assert any(
        "etapa=tarefa_inferencia_concluida" in msg
        and "status=ok" in msg
        and "tipo_erro=nenhum" in msg
        for msg in messages
    )


def test_response_treatment_failure_is_logged(monkeypatch, caplog):
    monkeypatch.setattr(
        main,
        "cliente",
        lambda: FakeClient(result=[]),
    )
    caplog.set_level(logging.INFO, logger="uvicorn.error")

    async def run():
        with pytest.raises(HTTPException) as caught:
            await main.analisar_imagem(b"fake-image")
        await asyncio.sleep(0)
        assert caught.value.status_code == 502

    asyncio.run(run())

    messages = diagnostic_messages(caplog)
    assert any(
        "etapa=tratamento_resposta" in msg
        and "status=erro" in msg
        and "tipo_erro=RuntimeError" in msg
        for msg in messages
    )

