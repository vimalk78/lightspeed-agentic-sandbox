"""Shared provider-egress TLS configuration and CA-bundle handling."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import ssl
import tempfile
from pathlib import Path
from typing import Any, Final, cast

import httpx

TLS_MOUNT_ROOT: Final = Path("/var/run/secrets/lightspeed/tls")
SERVICE_ACCOUNT_CA: Final = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")

_TLS_VERSION_NAMES: Final = {
    "VersionTLS10": ssl.TLSVersion.TLSv1,
    "VersionTLS11": ssl.TLSVersion.TLSv1_1,
    "VersionTLS12": ssl.TLSVersion.TLSv1_2,
    "VersionTLS13": ssl.TLSVersion.TLSv1_3,
}
_TLS13_CIPHER_NAMES: Final = frozenset(
    {
        "TLS_AES_128_CCM_SHA256",
        "TLS_AES_128_CCM_8_SHA256",
        "TLS_AES_128_GCM_SHA256",
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
    }
)
_LOGGER = logging.getLogger(__name__)

_CERTIFICATE_RE: Final = re.compile(
    rb"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", re.DOTALL
)


@dataclasses.dataclass(frozen=True)
class TLSConfig:
    """Operator-resolved TLS settings and the generated CA bundle path."""

    profile: str | None
    min_version: ssl.TLSVersion | None
    cipher_suites: tuple[str, ...]
    ca_bundle_path: Path | None = None
    ssl_context: ssl.SSLContext | None = None


_runtime_ssl_context: ssl.SSLContext | None = None


def get_ssl_context() -> ssl.SSLContext:
    """Return the startup TLS context, or platform defaults for direct use."""
    if _runtime_ssl_context is None:
        return ssl.create_default_context()
    return _runtime_ssl_context


def parse_tls_config() -> TLSConfig:
    """Parse operator-resolved TLS settings without selecting any defaults."""
    profile = os.environ.get("LIGHTSPEED_TLS_PROFILE", "").strip() or None
    min_version_name = os.environ.get("LIGHTSPEED_TLS_MIN_VERSION", "").strip() or None
    min_version = None
    if min_version_name is not None:
        try:
            min_version = _TLS_VERSION_NAMES[min_version_name]
        except KeyError as exc:
            supported = ", ".join(_TLS_VERSION_NAMES)
            raise ValueError(
                f"Unsupported LIGHTSPEED_TLS_MIN_VERSION {min_version_name!r}; "
                f"expected one of: {supported}"
            ) from exc

    cipher_value = os.environ.get("LIGHTSPEED_TLS_CIPHER_SUITES", "").strip()
    cipher_suites: tuple[str, ...] = ()
    if cipher_value:
        try:
            parsed = json.loads(cipher_value)
        except json.JSONDecodeError as exc:
            raise ValueError("LIGHTSPEED_TLS_CIPHER_SUITES must be valid JSON") from exc
        if not isinstance(parsed, list) or not all(isinstance(cipher, str) for cipher in parsed):
            raise ValueError("LIGHTSPEED_TLS_CIPHER_SUITES must be a JSON array of strings")
        cipher_suites = tuple(parsed)

    return TLSConfig(profile, min_version, cipher_suites)


def discover_ca_files(root: Path = TLS_MOUNT_ROOT) -> list[Path]:
    """Return regular mounted .crt and .pem files in deterministic order."""
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".crt", ".pem"}
    )


def _validate_certificate_file(path: Path) -> bytes:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Unable to read CA certificate {path}: {exc}") from exc

    certificates = _CERTIFICATE_RE.findall(content)
    if not certificates:
        raise ValueError(f"CA file {path} does not contain a PEM certificate")

    for certificate in certificates:
        try:
            ssl.PEM_cert_to_DER_cert(certificate.decode("ascii"))
        except (UnicodeDecodeError, ValueError, ssl.SSLError) as exc:
            raise ValueError(f"Malformed CA certificate in {path}") from exc
    return content


def _system_ca_bundle() -> bytes:
    cafile = ssl.get_default_verify_paths().cafile
    if cafile:
        path = Path(cafile)
        try:
            return path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Unable to read system CA bundle {path}: {exc}") from exc

    context = ssl.create_default_context()
    certificates = context.get_ca_certs(binary_form=True)
    if not certificates:
        raise ValueError("Unable to locate the platform/system CA bundle")
    return b"".join(ssl.DER_cert_to_PEM_cert(cert).encode("ascii") for cert in certificates)


def build_ca_bundle(
    root: Path = TLS_MOUNT_ROOT,
    output_path: Path | None = None,
) -> Path | None:
    """Create a bundle containing system, pod, and mounted CA certificates.

    The Kubernetes service-account CA is included when running in a pod. When
    neither it nor operator-provided CA files are present, this returns None to
    preserve normal local provider trust behavior. Source files and the system
    bundle are never modified.
    """
    ca_files = discover_ca_files(root)
    pod_ca = SERVICE_ACCOUNT_CA if SERVICE_ACCOUNT_CA.is_file() else None
    if not ca_files and pod_ca is None:
        return None

    mounted = [_validate_certificate_file(path) for path in ca_files]
    if pod_ca is not None:
        mounted.insert(0, _validate_certificate_file(pod_ca))
    system_bundle = _system_ca_bundle()
    if output_path is None:
        fd, generated_path = tempfile.mkstemp(prefix="lightspeed-ca-", suffix=".pem")
        os.close(fd)
        output_path = Path(generated_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_bytes(system_bundle + b"\n" + b"\n".join(mounted))
    except OSError as exc:
        raise ValueError(f"Unable to write combined CA bundle {output_path}: {exc}") from exc
    return output_path


def configure_tls(
    root: Path = TLS_MOUNT_ROOT,
    output_path: Path | None = None,
) -> TLSConfig:
    """Configure shared Python, gRPC, and AWS TLS settings."""
    global _runtime_ssl_context

    config = dataclasses.replace(
        parse_tls_config(),
        ca_bundle_path=build_ca_bundle(root, output_path),
    )
    context = create_ssl_context(config)
    config = dataclasses.replace(config, ssl_context=context)
    _runtime_ssl_context = context

    if config.ca_bundle_path is not None:
        bundle_path = str(config.ca_bundle_path)
        os.environ["SSL_CERT_FILE"] = bundle_path
        os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = bundle_path
        os.environ["AWS_CA_BUNDLE"] = bundle_path
    if config.cipher_suites:
        os.environ["GRPC_SSL_CIPHER_SUITES"] = ":".join(config.cipher_suites)
    return config


def create_http_client(
    *,
    httpx_module: Any | None = None,
    **kwargs: Any,
) -> httpx.Client:
    """Create a synchronous HTTP client using the shared TLS context."""
    client_module = httpx_module or httpx
    return cast(httpx.Client, client_module.Client(verify=get_ssl_context(), **kwargs))


def create_async_http_client(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
    *,
    httpx_module: Any | None = None,
    **kwargs: Any,
) -> httpx.AsyncClient:
    """Create an asynchronous HTTP client using the shared TLS context."""
    client_module = httpx_module or httpx
    return cast(
        httpx.AsyncClient,
        client_module.AsyncClient(
            verify=get_ssl_context(),
            headers=headers,
            timeout=timeout,
            auth=auth,
            **kwargs,
        ),
    )


def create_ssl_context(config: TLSConfig) -> ssl.SSLContext:
    """Create a certificate-verifying context from resolved TLS settings."""
    context = (
        ssl.create_default_context(cafile=str(config.ca_bundle_path))
        if config.ca_bundle_path is not None
        else ssl.create_default_context()
    )
    if config.min_version is not None:
        context.minimum_version = config.min_version

    tls13_ciphers = [cipher for cipher in config.cipher_suites if cipher in _TLS13_CIPHER_NAMES]
    legacy_ciphers = [
        cipher for cipher in config.cipher_suites if cipher not in _TLS13_CIPHER_NAMES
    ]
    if legacy_ciphers:
        try:
            context.set_ciphers(":".join(legacy_ciphers))
        except ssl.SSLError as exc:
            raise ValueError("Invalid TLS cipher suite configuration") from exc
        _LOGGER.info("Applied TLS 1.2-and-earlier cipher suites: %s", legacy_ciphers)

    if tls13_ciphers:
        set_ciphersuites = getattr(context, "set_ciphersuites", None)
        if set_ciphersuites is None:
            _LOGGER.warning(
                "TLS 1.3 cipher suites cannot be configured by this Python SSL runtime; "
                "leaving the runtime TLS 1.3 defaults enabled: %s",
                tls13_ciphers,
            )
        else:
            try:
                set_ciphersuites(":".join(tls13_ciphers))
            except (ssl.SSLError, ValueError) as exc:
                raise ValueError("Invalid TLS 1.3 cipher suite configuration") from exc
            _LOGGER.info("Applied TLS 1.3 cipher suites: %s", tls13_ciphers)
    return context
