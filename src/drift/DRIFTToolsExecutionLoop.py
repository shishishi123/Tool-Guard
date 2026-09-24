from .import_lib import *

class DRIFTToolsExecutionLoop(ToolsExecutionLoop):
    """Executes in loop a sequence of pipeline elements related to tool execution until the
    LLM does not return any tool calls.

    Args:
        elements: a sequence of pipeline elements to be executed in loop. One of them should be
            an LLM, and one of them should be a [ToolsExecutor][agentdojo.agent_pipeline.ToolsExecutor] (or
            something that behaves similarly by executing function calls). You can find an example usage
            of this class [here](../../concepts/agent_pipeline.md#combining-pipeline-components).
        max_iters: maximum number of iterations to execute the pipeline elements in loop.
    """

    def __init__(self, elements: Sequence[BasePipelineElement], max_iters: int = 15) -> None:
        self.elements = elements
        self.max_iters = max_iters

    def _ensure_string_content(self, content):
        """Convert content to string if it's a list (for AgentDojo compatibility)."""
        if isinstance(content, list):
            return " ".join(
                item.get("content", str(item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content) if content is not None else ""

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if len(messages) == 0:
            raise ValueError("Messages should not be empty when calling ToolsExecutionLoop")

        logger = Logger().get()
        for _ in range(self.max_iters):
            last_message = messages[-1]
            # Handle AgentDojo's list content format
            content_str = self._ensure_string_content(last_message.get("content", ""))
            if "[CALL ERROR]" not in content_str:
                if not last_message["role"] == "assistant":
                    break
                if last_message["tool_calls"] is None:
                    break
                if len(last_message["tool_calls"]) == 0:
                    break
            for element in self.elements:
                query, runtime, env, messages, extra_args = element.query(query, runtime, env, messages, extra_args)
                logger.log(messages)
        return query, runtime, env, messages, extra_args