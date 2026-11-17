"""Arranca el servidor.

    python run.py              # HTTP en el puerto 8000 (subir fotos desde la galería)
    python run.py --https      # HTTPS con certificado local (necesario para la cámara en vivo)
"""
from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import os
import secrets
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no envía nada; solo elige la interfaz
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def ensure_cert(cert_dir: Path, ip: str) -> tuple[Path, Path]:
    """Certificado autofirmado para localhost + IP de la red local."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    cert_dir.mkdir(parents=True, exist_ok=True)
    crt, key = cert_dir / f"cert-{ip}.pem", cert_dir / f"key-{ip}.pem"
    if crt.exists() and key.exists():
        return crt, key
    pk = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "scanner3d.local")])
    now = dt.datetime.now(dt.timezone.utc)
    san = x509.SubjectAlternativeName([
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        x509.IPAddress(ipaddress.ip_address(ip)),
    ])
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(pk.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=825))
            .add_extension(san, critical=False).sign(pk, hashes.SHA256()))
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key.write_bytes(pk.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                     serialization.NoEncryption()))
    return crt, key


def print_qr(url: str) -> None:
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.print_ascii(invert=True)
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--https", action="store_true", help="HTTPS con certificado autofirmado")
    ap.add_argument("--token", default=None, help="token de acceso (por defecto uno aleatorio)")
    ap.add_argument("--no-token", action="store_true", help="sin token (solo en redes de confianza)")
    ap.add_argument("--data", default=str(ROOT / "data"))
    a = ap.parse_args()

    import uvicorn

    token = "" if a.no_token else (a.token or os.environ.get("SCANNER_TOKEN") or secrets.token_urlsafe(9))
    os.environ["SCANNER_TOKEN"] = token
    os.environ["SCANNER_DATA"] = a.data
    ip = lan_ip()
    scheme = "https" if a.https else "http"
    url = f"{scheme}://{ip}:{a.port}/" + (f"?token={token}" if token else "")
    ssl = {}
    if a.https:
        crt, key = ensure_cert(Path(a.data) / "certs", ip)
        ssl = dict(ssl_certfile=str(crt), ssl_keyfile=str(key))

    print("\n  Scanner3D")
    print(f"  Abre en el celular (misma red Wi-Fi): {url}")
    if a.https:
        print("  El certificado es autofirmado: acepta la advertencia del navegador la primera vez.")
    else:
        print("  Sin HTTPS el navegador del celular no deja usar la cámara en vivo (solo galería).")
    print_qr(url)
    uvicorn.run("scanner3d.app:create_app", factory=True, host=a.host, port=a.port, **ssl)


if __name__ == "__main__":
    main()
