"""
05_upload_minio.py
==================
Envia as três camadas do Data Lake (bronze/prata/ouro) e o Data Warehouse para
um bucket S3-compatível no MinIO.

Pré-requisitos:
  - MinIO rodando (ver docker-compose.yml na raiz do projeto): `docker compose up -d`
  - pip install boto3
  - Variáveis de ambiente (ou os defaults abaixo, que batem com o docker-compose):
      MINIO_ENDPOINT   (default http://localhost:9000)
      MINIO_ACCESS_KEY (default minioadmin)
      MINIO_SECRET_KEY (default minioadmin)
      MINIO_BUCKET     (default datalake-huol)

Estrutura criada no bucket:
  datalake-huol/
    bronze/ ...   prata/ ...   ouro/ ...   dw/ ...

Uso:  python 05_upload_minio.py
"""

import os
import glob

try:
    import boto3
    from botocore.client import Config
except ImportError:
    raise SystemExit("Instale o boto3:  pip install boto3")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
ACCESS = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
SECRET = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
BUCKET = os.environ.get("MINIO_BUCKET", "datalake-huol")


def cliente():
    return boto3.client(
        "s3", endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS, aws_secret_access_key=SECRET,
        config=Config(signature_version="s3v4"), region_name="us-east-1",
    )


def main():
    s3 = cliente()
    # cria o bucket se não existir
    existentes = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if BUCKET not in existentes:
        s3.create_bucket(Bucket=BUCKET)
        print(f"Bucket criado: {BUCKET}")

    enviados = 0
    for camada in ("bronze", "prata", "ouro", "dw"):
        pasta = os.path.join(DATA, camada)
        if not os.path.isdir(pasta):
            continue
        for arq in glob.glob(os.path.join(pasta, "*")):
            chave = f"{camada}/{os.path.basename(arq)}"
            s3.upload_file(arq, BUCKET, chave)
            print(f"  ↑ {chave}")
            enviados += 1
    print(f"\n{enviados} arquivos enviados para {ENDPOINT}/{BUCKET}")
    print(f"Console web do MinIO: http://localhost:9001  (login: {ACCESS})")


if __name__ == "__main__":
    main()
