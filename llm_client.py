"""OpenAI client wrappers and mocks for structured prompt workflows."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

import workspace_config

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Message:
    """Simple chat message compatible with the OpenAI Responses API."""

    role: str
    content: str

    def as_dict(self) -> Mapping[str, str]:
        return {"role": self.role, "content": self.content}


class LLMClient:
    """Client that requests structured outputs from OpenAI models."""

    def __init__(
        self,
        model: str,
        *,
        client: Optional[OpenAI] = None,
        debug_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ) -> None:
        api_key, org_id = _load_credentials()
        if client is None:
            client = OpenAI(api_key=api_key, organization=org_id)
        self._client = client
        self._model = model
        self._debug_dir = debug_dir
        if self._debug_dir:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
        if log_dir is None:
            log_dir = workspace_config.get_llm_log_dir() / "conversations"
        self._log_dir = log_dir
        if self._log_dir:
            self._log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self) -> str:
        return self._model

    def request_structured(
        self,
        messages: Sequence[Message],
        response_model: Type[T],
        *,
        max_output_tokens: Optional[int] = None,
        extra_input: Optional[Iterable[Mapping[str, object]]] = None,
        max_attempts: int = 3,
        retry_reminder: Optional[str] = None,
    ) -> T:
        """
        Send messages to the model and parse the result into the provided Pydantic model.

        Parameters
        ----------
        messages:
            Ordered list of chat messages (must include a system message).
        response_model:
            Pydantic BaseModel subclass describing the expected structured output.
        max_output_tokens:
            Optional cap for generation length.
        extra_input:
            Additional payload entries to append to the request (e.g., tool definitions).
        """
        last_exception: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            raw_response = None
            current_messages = list(messages)
            if attempt > 1 and retry_reminder:
                current_messages = list(messages) + [Message(role="user", content=retry_reminder)]
            try:
                input_payload: List[Mapping[str, object]] = [msg.as_dict() for msg in current_messages]
                if extra_input:
                    input_payload.extend(extra_input)
                response = self._client.responses.parse(
                    model=self._model,
                    input=input_payload,
                    text_format=response_model,
                    max_output_tokens=max_output_tokens,
                    reasoning={
                        "effort": "medium",
                    },
                    text={
                        "verbosity": "low"
                    }
                )
                parsed = response.output_parsed
                if not isinstance(parsed, response_model):
                    raise TypeError(
                        f"Parsed response is not of expected type {response_model.__name__}"
                    )
                raw_response = response.model_dump()
                self._log_interaction(current_messages, extra_input, response_model, parsed, raw_response)
                return parsed
            except Exception as exc:
                last_exception = exc
                if self._debug_dir and raw_response is not None:
                    timestamp = int(time.time() * 1000)
                    debug_path = self._debug_dir / f"{response_model.__name__}_{timestamp}.json"
                    with debug_path.open("w", encoding="utf-8") as fh:
                        json.dump(raw_response, fh, indent=2, ensure_ascii=False)
                self._log_interaction(current_messages, extra_input, response_model, None, raw_response, error=str(exc))
                if attempt < max_attempts:
                    time.sleep(1)
                else:
                    raise RuntimeError(
                        f"Failed to obtain structured response for {response_model.__name__}: {exc}"
                    ) from exc

    def _log_interaction(
        self,
        messages: Sequence[Message],
        extra_input: Optional[Iterable[Mapping[str, object]]],
        response_model: Type[T],
        parsed: Optional[T],
        raw_response: Optional[Mapping[str, object]],
        error: Optional[str] = None,
    ) -> None:
        if not self._log_dir:
            return
        timestamp = int(time.time() * 1000)
        log_path = self._log_dir / f"{timestamp}_{response_model.__name__}.json"
        try:
            entry: Dict[str, object] = {
                "timestamp": timestamp,
                "model": self._model,
                "response_model": response_model.__name__,
                "messages": [msg.as_dict() for msg in messages],
            }
            if extra_input:
                entry["extra_input"] = list(extra_input)
            if parsed is not None:
                if hasattr(parsed, "model_dump"):
                    entry["parsed_output"] = parsed.model_dump()
                else:
                    entry["parsed_output"] = parsed
            if raw_response is not None:
                entry["raw_response"] = raw_response
            if error:
                entry["error"] = error
            with log_path.open("w", encoding="utf-8") as fh:
                json.dump(entry, fh, indent=2, ensure_ascii=False)
        except Exception:
            # Swallow logging failures to avoid impacting main flow.
            pass


def _load_credentials() -> tuple[str, Optional[str]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    org_id = os.environ.get("OPENAI_ORG_ID")

    if not api_key:
        candidate = Path(__file__).resolve().parent.parent / "API_key.txt"
        if candidate.is_file():
            api_key = candidate.read_text(encoding="utf-8").strip()

    if not api_key:
        raise EnvironmentError(
            "OpenAI API key not found. Set OPENAI_API_KEY or provide ../API_key.txt."
        )

    return api_key, org_id



__all__ = ["LLMClient", "Message"]
