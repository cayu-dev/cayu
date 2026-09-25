"""The access wrapper preserves the actual generator owner's stream protocol."""

import asyncio

import pytest

from cayu.resource_access import _model_data_access, runtime_stream_entrance


@pytest.mark.parametrize("termination", ["complete", "close", "throw"])
def test_runtime_stream_forwards_injected_errors_without_leaking_access(termination):
    async def run():
        evidence = []
        original = RuntimeError("consumer failed")

        class Owner:
            @runtime_stream_entrance
            async def stream(self):
                try:
                    assert _model_data_access.get() is False
                    try:
                        yield "started"
                    except RuntimeError as error:
                        assert error is original
                        assert _model_data_access.get() is False
                        evidence.append(error)
                        yield "recovered"
                    assert _model_data_access.get() is False
                finally:
                    assert _model_data_access.get() is False
                    evidence.append("closed")

        token = _model_data_access.set(True)
        stream = Owner().stream()
        try:
            assert await anext(stream) == "started"
            assert _model_data_access.get() is True
            if termination == "throw":
                assert await stream.athrow(original) == "recovered"
                assert _model_data_access.get() is True
            if termination == "close":
                await stream.aclose()
            else:
                with pytest.raises(StopAsyncIteration):
                    await anext(stream)
            assert _model_data_access.get() is True
            assert evidence == ([original, "closed"] if termination == "throw" else ["closed"])
        finally:
            await stream.aclose()
            _model_data_access.reset(token)

    asyncio.run(run())
