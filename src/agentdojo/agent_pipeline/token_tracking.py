"""
Token Tracking Utility

Provides a unified way to track token usage across different LLM providers
(OpenAI, Anthropic, Google) during defense evaluation.

Usage:
    tracker = TokenTracker()
    
    # Wrap your client
    wrapped_client = tracker.wrap_openai_client(openai_client)
    
    # Use the wrapped client normally - tokens are tracked automatically
    response = wrapped_client.chat.completions.create(...)
    
    # Get stats
    stats = tracker.get_stats()
    print(f"Total tokens: {stats['total_tokens']}")
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    """Token usage for a single API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    source: str = ""  # e.g., "main_llm", "validator", "replan"


@dataclass
class TokenStats:
    """Aggregated token statistics."""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    usage_by_source: dict = field(default_factory=dict)
    usage_history: list = field(default_factory=list)
    
    def add_usage(self, usage: TokenUsage):
        """Add a usage record to the stats."""
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.call_count += 1
        
        # Track by source
        if usage.source not in self.usage_by_source:
            self.usage_by_source[usage.source] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "call_count": 0,
            }
        self.usage_by_source[usage.source]["prompt_tokens"] += usage.prompt_tokens
        self.usage_by_source[usage.source]["completion_tokens"] += usage.completion_tokens
        self.usage_by_source[usage.source]["total_tokens"] += usage.total_tokens
        self.usage_by_source[usage.source]["call_count"] += 1
        
        # Keep history
        self.usage_history.append(usage)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "usage_by_source": self.usage_by_source,
        }


class TokenTracker:
    """
    Unified token tracking across multiple LLM providers.
    
    This class provides wrappers for OpenAI, Anthropic, and Google clients
    that automatically track token usage from API responses.
    """
    
    def __init__(self):
        self.stats = TokenStats()
        self._current_source = "default"
    
    def set_source(self, source: str):
        """Set the current source label for tracking."""
        self._current_source = source
    
    def reset(self):
        """Reset all tracking stats."""
        self.stats = TokenStats()
        self._current_source = "default"
    
    def get_stats(self) -> dict:
        """Get current token statistics as a dictionary."""
        return self.stats.to_dict()
    
    def record_usage(self, prompt_tokens: int, completion_tokens: int, source: str = None):
        """Manually record token usage."""
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            source=source or self._current_source,
        )
        self.stats.add_usage(usage)
    
    # =========================================================================
    # OpenAI Wrapper
    # =========================================================================
    
    def wrap_openai_client(self, client, source: str = "openai"):
        """
        Wrap an OpenAI client to track token usage.
        
        Args:
            client: The OpenAI client instance
            source: Label for this client's usage
        
        Returns:
            A wrapped client that tracks tokens
        """
        return OpenAIClientWrapper(client, self, source)
    
    # =========================================================================
    # Anthropic Wrapper
    # =========================================================================
    
    def wrap_anthropic_client(self, client, source: str = "anthropic"):
        """
        Wrap an Anthropic client to track token usage.
        
        Args:
            client: The Anthropic client instance
            source: Label for this client's usage
        
        Returns:
            A wrapped client that tracks tokens
        """
        return AnthropicClientWrapper(client, self, source)
    
    # =========================================================================
    # Google Wrapper
    # =========================================================================
    
    def wrap_google_client(self, client, source: str = "google"):
        """
        Wrap a Google GenAI client to track token usage.
        
        Args:
            client: The Google GenAI client instance
            source: Label for this client's usage
        
        Returns:
            A wrapped client that tracks tokens
        """
        return GoogleClientWrapper(client, self, source)


# =============================================================================
# OpenAI Client Wrapper
# =============================================================================

class OpenAIClientWrapper:
    """Wrapper for OpenAI client that tracks token usage."""
    
    def __init__(self, client, tracker: TokenTracker, source: str):
        self._client = client
        self._tracker = tracker
        self._source = source
        self.chat = OpenAIChatWrapper(client.chat, tracker, source)
    
    def __getattr__(self, name):
        # Pass through other attributes to the underlying client
        return getattr(self._client, name)


class OpenAIChatWrapper:
    """Wrapper for OpenAI chat interface."""
    
    def __init__(self, chat, tracker: TokenTracker, source: str):
        self._chat = chat
        self._tracker = tracker
        self._source = source
        self.completions = OpenAICompletionsWrapper(chat.completions, tracker, source)
    
    def __getattr__(self, name):
        return getattr(self._chat, name)


class OpenAICompletionsWrapper:
    """Wrapper for OpenAI completions that captures token usage."""
    
    def __init__(self, completions, tracker: TokenTracker, source: str):
        self._completions = completions
        self._tracker = tracker
        self._source = source
    
    def create(self, *args, **kwargs):
        """Create a completion and track token usage."""
        response = self._completions.create(*args, **kwargs)
        
        # Extract usage from response
        if hasattr(response, 'usage') and response.usage:
            self._tracker.record_usage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                source=self._source,
            )
        
        return response
    
    def __getattr__(self, name):
        return getattr(self._completions, name)


# =============================================================================
# Anthropic Client Wrapper
# =============================================================================

class AnthropicClientWrapper:
    """Wrapper for Anthropic client that tracks token usage."""
    
    def __init__(self, client, tracker: TokenTracker, source: str):
        self._client = client
        self._tracker = tracker
        self._source = source
        self.messages = AnthropicMessagesWrapper(client.messages, tracker, source)
    
    def __getattr__(self, name):
        return getattr(self._client, name)


class AnthropicMessagesWrapper:
    """Wrapper for Anthropic messages interface."""
    
    def __init__(self, messages, tracker: TokenTracker, source: str):
        self._messages = messages
        self._tracker = tracker
        self._source = source
    
    def create(self, *args, **kwargs):
        """Create a message and track token usage."""
        response = self._messages.create(*args, **kwargs)
        
        # Extract usage from response
        if hasattr(response, 'usage') and response.usage:
            self._tracker.record_usage(
                prompt_tokens=response.usage.input_tokens or 0,
                completion_tokens=response.usage.output_tokens or 0,
                source=self._source,
            )
        
        return response
    
    def stream(self, *args, **kwargs):
        """Stream messages - wraps to capture final usage."""
        return AnthropicStreamWrapper(
            self._messages.stream(*args, **kwargs),
            self._tracker,
            self._source,
        )
    
    def __getattr__(self, name):
        return getattr(self._messages, name)


class AnthropicStreamWrapper:
    """Wrapper for Anthropic streaming that captures final usage."""
    
    def __init__(self, stream_context, tracker: TokenTracker, source: str):
        self._stream_context = stream_context
        self._tracker = tracker
        self._source = source
    
    def __enter__(self):
        self._stream = self._stream_context.__enter__()
        return AnthropicStreamInstanceWrapper(self._stream, self._tracker, self._source)
    
    def __exit__(self, *args):
        return self._stream_context.__exit__(*args)
    
    async def __aenter__(self):
        self._stream = await self._stream_context.__aenter__()
        return AnthropicStreamInstanceWrapper(self._stream, self._tracker, self._source)
    
    async def __aexit__(self, *args):
        return await self._stream_context.__aexit__(*args)


class AnthropicStreamInstanceWrapper:
    """Wrapper for individual Anthropic stream instance."""
    
    def __init__(self, stream, tracker: TokenTracker, source: str):
        self._stream = stream
        self._tracker = tracker
        self._source = source
    
    async def get_final_message(self):
        """Get final message and track usage."""
        response = await self._stream.get_final_message()
        
        if hasattr(response, 'usage') and response.usage:
            self._tracker.record_usage(
                prompt_tokens=response.usage.input_tokens or 0,
                completion_tokens=response.usage.output_tokens or 0,
                source=self._source,
            )
        
        return response
    
    def __getattr__(self, name):
        return getattr(self._stream, name)


# =============================================================================
# Google Client Wrapper
# =============================================================================

class GoogleClientWrapper:
    """Wrapper for Google GenAI client that tracks token usage."""
    
    def __init__(self, client, tracker: TokenTracker, source: str):
        self._client = client
        self._tracker = tracker
        self._source = source
        self.models = GoogleModelsWrapper(client.models, tracker, source)
    
    def __getattr__(self, name):
        return getattr(self._client, name)


class GoogleModelsWrapper:
    """Wrapper for Google models interface."""
    
    def __init__(self, models, tracker: TokenTracker, source: str):
        self._models = models
        self._tracker = tracker
        self._source = source
    
    def generate_content(self, *args, **kwargs):
        """Generate content and track token usage."""
        response = self._models.generate_content(*args, **kwargs)
        
        # Extract usage from response
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            self._tracker.record_usage(
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                completion_tokens=response.usage_metadata.candidates_token_count or 0,
                source=self._source,
            )
        
        return response
    
    def __getattr__(self, name):
        return getattr(self._models, name)


# =============================================================================
# Utility Functions
# =============================================================================

def create_tracked_openai_client(api_key: str = None, tracker: TokenTracker = None, source: str = "openai"):
    """
    Create an OpenAI client with token tracking enabled.
    
    Args:
        api_key: OpenAI API key (uses env var if not provided)
        tracker: TokenTracker instance (creates new one if not provided)
        source: Label for tracking
    
    Returns:
        Tuple of (wrapped_client, tracker)
    """
    import openai
    import os
    
    client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    tracker = tracker or TokenTracker()
    wrapped = tracker.wrap_openai_client(client, source)
    
    return wrapped, tracker


def create_tracked_anthropic_client(api_key: str = None, tracker: TokenTracker = None, source: str = "anthropic"):
    """
    Create an Anthropic client with token tracking enabled.
    
    Args:
        api_key: Anthropic API key (uses env var if not provided)
        tracker: TokenTracker instance (creates new one if not provided)
        source: Label for tracking
    
    Returns:
        Tuple of (wrapped_client, tracker)
    """
    from anthropic import Anthropic
    import os
    
    client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    tracker = tracker or TokenTracker()
    wrapped = tracker.wrap_anthropic_client(client, source)
    
    return wrapped, tracker


def create_tracked_google_client(api_key: str = None, tracker: TokenTracker = None, source: str = "google"):
    """
    Create a Google GenAI client with token tracking enabled.
    
    Args:
        api_key: Google API key (uses env var if not provided)
        tracker: TokenTracker instance (creates new one if not provided)
        source: Label for tracking
    
    Returns:
        Tuple of (wrapped_client, tracker)
    """
    from google import genai
    import os
    
    client = genai.Client(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
    tracker = tracker or TokenTracker()
    wrapped = tracker.wrap_google_client(client, source)
    
    return wrapped, tracker

