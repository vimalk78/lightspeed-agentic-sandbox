"""Tests for shared provider-egress TLS configuration."""

from __future__ import annotations

import json
import os
import re
import ssl
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest

from lightspeed_agentic import tls  # type: ignore[import-untyped]

_CERTIFICATE_RE = re.compile(rb"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", re.DOTALL)


@pytest.fixture(autouse=True)
def _restore_tls_process_state() -> Iterator[None]:
    env_names = (
        "SSL_CERT_FILE",
        "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH",
        "AWS_CA_BUNDLE",
        "GRPC_SSL_CIPHER_SUITES",
    )
    previous_env = {name: os.environ.get(name) for name in env_names}
    previous_context = tls._runtime_ssl_context

    yield

    for name, value in previous_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    tls._runtime_ssl_context = previous_context


def _system_certificate() -> bytes:
    cafile = ssl.get_default_verify_paths().cafile
    if not cafile:
        pytest.skip("system CA bundle is unavailable")
    matches = _CERTIFICATE_RE.findall(Path(cafile).read_bytes())
    if not matches:
        pytest.skip("system CA bundle contains no PEM certificates")
    return bytes(matches[0]) + b"\n"


def test_parse_tls_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_TLS_PROFILE", "IntermediateType")
    monkeypatch.setenv("LIGHTSPEED_TLS_MIN_VERSION", "VersionTLS12")
    monkeypatch.setenv(
        "LIGHTSPEED_TLS_CIPHER_SUITES",
        json.dumps(["ECDHE-RSA-AES128-GCM-SHA256"]),
    )

    config = tls.parse_tls_config()

    assert config.profile == "IntermediateType"
    assert config.min_version == ssl.TLSVersion.TLSv1_2
    assert config.cipher_suites == ("ECDHE-RSA-AES128-GCM-SHA256",)
    assert config.ca_bundle_path is None


def test_parse_tls_config_rejects_invalid_cipher_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIGHTSPEED_TLS_CIPHER_SUITES", "not-json")

    with pytest.raises(ValueError, match="must be valid JSON"):
        tls.parse_tls_config()


def test_discover_ca_files_recursively_and_ignores_other_extensions(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "integration" / "nested"
    nested.mkdir(parents=True)
    (tmp_path / "root.CRT").write_text("certificate")
    (nested / "extra.pem").write_text("certificate")
    (nested / "ignored.key").write_text("key")

    assert tls.discover_ca_files(tmp_path) == [
        nested / "extra.pem",
        tmp_path / "root.CRT",
    ]


def test_build_ca_bundle_preserves_system_roots_and_validates_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mounted = tmp_path / "mounted" / "additional.crt"
    mounted.parent.mkdir()
    certificate = _system_certificate()
    mounted.write_bytes(certificate)
    output = tmp_path / "generated" / "bundle.pem"
    monkeypatch.setattr(tls, "_system_ca_bundle", lambda: b"SYSTEM-ROOTS")
    monkeypatch.setattr(tls, "SERVICE_ACCOUNT_CA", tmp_path / "missing-service-account-ca.crt")

    bundle_path = tls.build_ca_bundle(mounted.parent.parent, output)

    assert bundle_path == output
    assert output.read_bytes() == b"SYSTEM-ROOTS\n" + certificate
    assert mounted.read_bytes() == certificate


def test_build_ca_bundle_includes_service_account_ca(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pod_ca = tmp_path / "serviceaccount-ca.crt"
    certificate = _system_certificate()
    pod_ca.write_bytes(certificate)
    monkeypatch.setattr(tls, "SERVICE_ACCOUNT_CA", pod_ca)
    monkeypatch.setattr(tls, "_system_ca_bundle", lambda: b"SYSTEM-ROOTS")

    bundle_path = tls.build_ca_bundle(tmp_path / "empty", tmp_path / "bundle.pem")

    assert bundle_path is not None
    assert bundle_path.read_bytes() == b"SYSTEM-ROOTS\n" + certificate


def test_build_ca_bundle_includes_all_mounted_ca_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = ["otel", "mcp", "rhokp", "additional-ca"]
    certificate = _system_certificate()
    for source in sources:
        source_dir = tmp_path / source
        source_dir.mkdir()
        (source_dir / f"{source}.crt").write_bytes(certificate)
    monkeypatch.setattr(tls, "_system_ca_bundle", lambda: b"SYSTEM-ROOTS")

    bundle_path = tls.build_ca_bundle(tmp_path, tmp_path / "bundle.pem")

    assert bundle_path is not None
    bundle = bundle_path.read_bytes()
    assert bundle.startswith(b"SYSTEM-ROOTS\n")
    assert bundle.count(certificate) == len(sources)


def test_build_ca_bundle_rejects_malformed_certificate(tmp_path: Path) -> None:
    mounted = tmp_path / "broken.pem"
    mounted.write_text("-----BEGIN CERTIFICATE-----\ninvalid\n-----END CERTIFICATE-----\n")

    with pytest.raises(ValueError, match="Malformed CA certificate"):
        tls.build_ca_bundle(tmp_path, tmp_path / "bundle.pem")


def test_configure_tls_sets_https_and_grpc_bundle_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mounted = tmp_path / "mounted" / "ca.crt"
    mounted.parent.mkdir()
    mounted.write_bytes(_system_certificate())
    output = tmp_path / "bundle.pem"
    monkeypatch.setattr(tls, "_system_ca_bundle", lambda: b"SYSTEM-ROOTS")
    monkeypatch.setenv(
        "LIGHTSPEED_TLS_CIPHER_SUITES",
        json.dumps(["ECDHE-RSA-AES128-GCM-SHA256", "ECDHE-RSA-AES256-GCM-SHA384"]),
    )

    config = tls.configure_tls(mounted.parent.parent, output)

    assert config.ca_bundle_path == output
    assert os.environ["SSL_CERT_FILE"] == str(output)
    assert os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] == str(output)
    assert os.environ["AWS_CA_BUNDLE"] == str(output)
    assert os.environ["GRPC_SSL_CIPHER_SUITES"] == (
        "ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384"
    )
    assert tls.get_ssl_context() is config.ssl_context


def test_configure_tls_creates_shared_context_with_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mounted = tmp_path / "mounted" / "ca.crt"
    mounted.parent.mkdir()
    mounted.write_bytes(_system_certificate())
    monkeypatch.setattr(tls, "_system_ca_bundle", lambda: b"SYSTEM-ROOTS")
    monkeypatch.setenv("LIGHTSPEED_TLS_MIN_VERSION", "VersionTLS13")
    monkeypatch.setenv(
        "LIGHTSPEED_TLS_CIPHER_SUITES",
        json.dumps(["ECDHE-RSA-AES256-GCM-SHA384"]),
    )

    config = tls.configure_tls(mounted.parent.parent, tmp_path / "bundle.pem")

    assert config.ssl_context is not None
    assert config.ssl_context is tls.get_ssl_context()
    assert config.ssl_context.minimum_version == ssl.TLSVersion.TLSv1_3
    assert config.ssl_context.get_ciphers()


def test_httpx_factories_use_shared_ssl_context(monkeypatch: pytest.MonkeyPatch) -> None:
    context = ssl.create_default_context()
    monkeypatch.setattr(tls, "get_ssl_context", lambda: context)

    sync_client = tls.create_http_client()
    async_client = tls.create_async_http_client()
    try:
        assert cast(Any, sync_client._transport)._pool._ssl_context is context
        assert cast(Any, async_client._transport)._pool._ssl_context is context
    finally:
        sync_client.close()
        awaitable = async_client.aclose()
        import asyncio

        asyncio.run(awaitable)


def test_httpx_factories_use_supplied_module() -> None:
    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeAsyncClient(FakeClient):
        pass

    class FakeHttpx:
        Client = FakeClient
        AsyncClient = FakeAsyncClient

    sync_client = tls.create_http_client(httpx_module=FakeHttpx)
    async_client = tls.create_async_http_client(httpx_module=FakeHttpx)

    assert isinstance(sync_client, FakeClient)
    assert isinstance(async_client, FakeAsyncClient)


def test_create_ssl_context_applies_tls13_ciphersuites_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = Mock()
    context.set_ciphersuites = Mock()
    monkeypatch.setattr(tls.ssl, "create_default_context", lambda **_: context)
    config = tls.TLSConfig(
        None,
        ssl.TLSVersion.TLSv1_2,
        ("ECDHE-RSA-AES128-GCM-SHA256", "TLS_AES_128_GCM_SHA256"),
        None,
    )

    tls.create_ssl_context(config)

    context.set_ciphers.assert_called_once_with("ECDHE-RSA-AES128-GCM-SHA256")
    context.set_ciphersuites.assert_called_once_with("TLS_AES_128_GCM_SHA256")


def test_create_ssl_context_rejects_invalid_tls13_ciphersuites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = Mock()
    context.set_ciphersuites.side_effect = ssl.SSLError("invalid cipher suite")
    monkeypatch.setattr(tls.ssl, "create_default_context", lambda **_: context)
    config = tls.TLSConfig(None, None, ("TLS_AES_128_GCM_SHA256",), None)

    with pytest.raises(ValueError, match=r"Invalid TLS 1\.3 cipher suite"):
        tls.create_ssl_context(config)


def test_create_ssl_context_warns_when_tls13_cipher_configuration_is_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = tls.TLSConfig(None, ssl.TLSVersion.TLSv1_3, ("TLS_AES_128_GCM_SHA256",), None)

    with caplog.at_level("WARNING", logger="lightspeed_agentic.tls"):
        tls.create_ssl_context(config)

    assert "TLS 1.3 cipher suites cannot be configured" in caplog.text


def test_create_ssl_context_applies_minimum_version() -> None:
    config = tls.TLSConfig(None, ssl.TLSVersion.TLSv1_2, (), None)

    context = tls.create_ssl_context(config)

    assert context.minimum_version == ssl.TLSVersion.TLSv1_2
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
