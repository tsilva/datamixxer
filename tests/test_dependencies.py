from __future__ import annotations

import asyncio
from importlib.metadata import version
from time import perf_counter

import aiohttp
import idna
import pytest
from packaging.version import Version


async def _serve_once(response: bytes) -> tuple[asyncio.AbstractServer, str]:
    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(response)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"http://127.0.0.1:{port}/"


def test_dependency_versions_include_security_fixes() -> None:
    assert Version(version("aiohttp")) >= Version("3.14.3")
    assert Version(version("idna")) >= Version("3.15")


def test_malformed_chunked_response_is_rejected_without_parser_crash() -> None:
    async def scenario() -> None:
        response = (
            b"HTTP/1.1 200 OK\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Connection: close\r\n\r\n"
            b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF\r\ninvalid\r\n0\r\n\r\n"
        )
        server, url = await _serve_once(response)
        try:
            async with aiohttp.ClientSession() as session:
                with pytest.raises((aiohttp.ClientPayloadError, aiohttp.ClientResponseError)):
                    async with session.get(url) as result:
                        await result.read()
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_valid_chunked_response_still_round_trips() -> None:
    async def scenario() -> None:
        response = (
            b"HTTP/1.1 200 OK\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Connection: close\r\n\r\n"
            b"5\r\nhello\r\n0\r\n\r\n"
        )
        server, url = await _serve_once(response)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as result:
                    assert await result.text() == "hello"
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_oversized_contextual_idna_label_is_rejected_early() -> None:
    started = perf_counter()
    with pytest.raises(idna.IDNAError):
        idna.alabel("\u0660" * 100_000)
    assert perf_counter() - started < 0.5

    assert idna.encode("m\u00fcnich.example") == b"xn--mnich-kva.example"
