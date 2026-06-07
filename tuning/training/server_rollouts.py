# ABOUTME: Adapts TRL server-mode chat rollouts to client-rendered text prompts.
# ABOUTME: Avoids relying on the vLLM sidecar's tokenizer or saved chat template.

from __future__ import annotations

from trl.data_utils import apply_chat_template


def _contains_images(messages: list[list[dict]]) -> bool:
    for conversation in messages:
        for message in conversation:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"image", "image_pil", "image_url"}:
                    return True
                if any(key in part for key in ("image", "image_pil", "image_url")):
                    return True
    return False


def install_client_rendered_chat(client, processing_class) -> None:
    """Render conversations locally and route TRL's chat calls through /generate/."""
    if client.__dict__.get("_client_rendered_chat_installed", False):
        return

    def rendered_chat(
        messages,
        n=1,
        repetition_penalty=1.0,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        max_tokens=16,
        logprobs=0,
        truncate_prompt_tokens=None,
        structured_outputs_regex=None,
        generation_kwargs=None,
        chat_template_kwargs=None,
        tools=None,
        chat_template=None,
    ):
        if _contains_images(messages):
            raise NotImplementedError(
                "Client-rendered GRPO server rollouts currently support text-only prompts."
            )

        template_kwargs = dict(chat_template_kwargs or {})
        if chat_template is not None:
            template_kwargs["chat_template"] = chat_template
        prompts = [
            apply_chat_template(
                {"prompt": conversation},
                processing_class,
                tools=tools,
                **template_kwargs,
            )["prompt"]
            for conversation in messages
        ]
        return client.generate(
            prompts=prompts,
            n=n,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
            logprobs=logprobs,
            truncate_prompt_tokens=truncate_prompt_tokens,
            structured_outputs_regex=structured_outputs_regex,
            generation_kwargs=generation_kwargs,
        )

    client.chat = rendered_chat
    client._client_rendered_chat_installed = True
