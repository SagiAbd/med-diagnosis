from minio import Minio
from app.core.config import settings

def get_minio_client() -> Minio:
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=False  # Set to True if using HTTPS
    )

def init_minio():
    client = get_minio_client()
    # Create bucket if it doesn't exist
    if not client.bucket_exists(settings.MINIO_BUCKET_NAME):
        client.make_bucket(settings.MINIO_BUCKET_NAME) 