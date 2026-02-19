from pydantic import BaseModel, model_validator
from typing import Optional


class VLLMSamplingParamsConfig(BaseModel):
    max_tokens: int = 4096
    temperature: float = 0.5
    top_k: int = 150
    top_p: float = 0.9
    stop: list[str] = []
    chat_template: Optional[str] = None  # None = read from global at init time
    # repetition_penalty: float = 1.1
    n: int = 1

    @model_validator(mode="after")
    def _resolve_stop_tokens(self):
        from tuning.config import DEFAULT_CHAT_TEMPLATE  # lazy import = runtime read
        from tuning.utils.utils import get_stop_tokens
        if self.chat_template is None:
            self.chat_template = DEFAULT_CHAT_TEMPLATE
        if not self.stop:
            self.stop = get_stop_tokens(self.chat_template)
        return self

    def model_dump(self, **kwargs):
        d = super().model_dump(**kwargs)
        d.pop("chat_template", None)  # exclude from SamplingParams(**config.model_dump())
        return d


if __name__ == "__main__":
    print({**VLLMSamplingParamsConfig().model_dump()})
