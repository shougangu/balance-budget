from typing import Optional

from pydantic import BaseModel, model_validator


class VLLMSamplingParamsConfig(BaseModel):
    max_tokens: int = 4096
    temperature: float = 0.5
    top_k: int = 150
    top_p: float = 0.9
    stop: list[str] = []
    # stop_token_ids: list[int] = [128009, 128001]
    # repetition_penalty: float = 1.1
    n: int = 1
    seed: Optional[int] = None

    @model_validator(mode="after")
    def _resolve_defaults(self):
        from tuning.utils.utils import get_stop_tokens
        import tuning.config
        if not self.stop:
            self.stop = get_stop_tokens()
        if self.seed is None:
            self.seed = tuning.config.get_eval_seed()
        return self

if __name__ == "__main__":
    print({**VLLMSamplingParamsConfig().model_dump()})
