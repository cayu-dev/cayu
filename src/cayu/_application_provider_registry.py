"""Provider declarations, default selection and model-pattern routing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from fnmatch import fnmatchcase
from types import MappingProxyType

from cayu._application_registration import _validate_provider_model_patterns
from cayu._validation import require_clean_nonblank
from cayu.providers.base import ModelProvider, copy_usage_dialect
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._execution_profile_identity_validation import (
    copy_secret_free_execution_profile_behavior_identity,
)
from cayu.vaults.redaction import SecretRedactor


class ApplicationProviderRegistry:
    """Own validated provider declarations and their default without an application.

    Registration snapshots provider identity, usage dialect and routing patterns.
    The live provider remains available for execution. Callers supply provenance
    captured at their public registration boundary.
    """

    def __init__(self, *, secret_redactor: SecretRedactor) -> None:
        self._secret_redactor = secret_redactor
        self._providers: dict[str, runtime_records.RegisteredProvider] = {}
        self._default_name: str | None = None

    @property
    def registrations(self) -> Mapping[str, runtime_records.RegisteredProvider]:
        """Read current declarations without exposing registry mutation."""
        return MappingProxyType(self._providers)

    @property
    def default_name(self) -> str | None:
        return self._default_name

    def register(
        self,
        provider: ModelProvider,
        *,
        default: bool = False,
        model_patterns: Iterable[str] | None = None,
        registration_site: tuple[str | None, str | None] = (None, None),
    ) -> ModelProvider:
        if not isinstance(provider, ModelProvider):
            raise TypeError("Provider registration requires a ModelProvider.")
        if not isinstance(default, bool):
            raise TypeError("Provider default flag must be a bool.")
        stored_model_patterns = _validate_provider_model_patterns(model_patterns)
        provider_name = require_clean_nonblank(provider.name, "provider.name")
        usage_dialect = copy_usage_dialect(provider.usage_dialect, "provider.usage_dialect")
        if provider_name in self._providers:
            raise ValueError(f"Provider already registered: {provider_name}")

        registration_source, registration_symbol = registration_site
        self._providers[provider_name] = runtime_records.RegisteredProvider(
            name=provider_name,
            provider=provider,
            execution_profile_identity=(
                copy_secret_free_execution_profile_behavior_identity(
                    provider.execution_profile_identity,
                    redactor=self._secret_redactor,
                    field_name="provider.execution_profile_identity",
                )
            ),
            model_patterns=stored_model_patterns,
            registration_source=registration_source,
            registration_symbol=registration_symbol,
            usage_dialect=usage_dialect,
        )
        if default or self._default_name is None:
            self._default_name = provider_name
        return provider

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def get(self, name: str | None = None) -> runtime_records.RegisteredProvider:
        if name is not None:
            provider_name = require_clean_nonblank(name, "provider.name")
        else:
            provider_name = self._default_name
        if provider_name is None:
            raise RuntimeError("No model provider registered.")
        try:
            return self._providers[provider_name]
        except KeyError as exc:
            raise KeyError(f"Provider not registered: {provider_name}") from exc

    def matching(self, *, model: str) -> tuple[runtime_records.RegisteredProvider, ...]:
        """Find pattern matches in registration order, including ambiguous matches."""
        model = require_clean_nonblank(model, "model")
        return tuple(
            provider
            for provider in self._providers.values()
            if any(fnmatchcase(model, pattern) for pattern in provider.model_patterns)
        )

    def route(self, *, model: str) -> runtime_records.RegisteredProvider | None:
        matches = self.matching(model=model)
        if not matches:
            return None
        if len(matches) > 1:
            match_names = ", ".join(provider.name for provider in matches)
            raise ValueError(
                f"Model matches multiple registered providers: {model} -> {match_names}"
            )
        return matches[0]

    def replace_for_replay(
        self,
        registrations: Mapping[str, runtime_records.RegisteredProvider],
        *,
        default_name: str | None,
    ) -> None:
        """Install isolated replay records while retaining their admitted identities.

        Replay substitutes recorded providers in existing validated declarations;
        registering those providers again would snapshot their wrapper identities.
        Copy the mapping so later source-map mutations cannot alter this registry.
        """
        providers = dict(registrations)
        if default_name is not None and default_name not in providers:
            raise ValueError(f"Replay default provider not registered: {default_name}")
        self._providers = providers
        self._default_name = default_name
