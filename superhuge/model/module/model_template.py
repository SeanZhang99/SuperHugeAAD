from pydantic import BaseModel


class ModelInputArgs(BaseModel):
    fs: int
    window_length: int
    num_channels: int | None = None
    num_audio_features: int | None = None

    model_config = {"extra": "allow"}
