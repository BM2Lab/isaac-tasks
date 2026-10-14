import pytest
from contracts.actions import ActionChunk, CartesianDeltaSpec
from contracts.episodes import SampleContext
from types import SimpleNamespace
from isaac_tasks.bridges.action_queue import ActionQueue


def setup_queue():
    action = CartesianDeltaSpec("arm", "base", "tip", 20, .005)
    queue = ActionQueue(SimpleNamespace(action=action, chunk_size=50))
    context = SampleContext("episode", 0, 0, "simulation")
    chunk = ActionChunk(context, action, ((.001, 0., 0.), (0., 0., 0.)), 100_000_000)
    return queue, context, chunk


def test_ordered_chunk_consumption():
    queue, context, chunk = setup_queue()
    queue.accept(chunk, context)
    assert queue.pop(context) == chunk.values[0]
    assert queue.pop(SampleContext("episode", 1, 50_000_000, "simulation")) == chunk.values[1]
    with pytest.raises(ValueError, match="No queued"):
        queue.pop(SampleContext("episode", 2, 100_000_000, "simulation"))


def test_clear_and_old_context_rejected():
    queue, context, chunk = setup_queue()
    queue.accept(chunk, context)
    queue.clear()
    assert len(queue) == 0
    with pytest.raises(ValueError):
        queue.accept(chunk, SampleContext("next", 0, 50_000_000, "simulation"))
    with pytest.raises(ValueError):
        queue.accept(chunk, SampleContext("episode", 0, 1, "simulation"))


@pytest.mark.parametrize("context", [SampleContext("next", 0, 0, "simulation"),
    SampleContext("episode", 1, 50_000_000, "simulation"), SampleContext("episode", 0, 100_000_000, "simulation"),
    SampleContext("episode", 0, 0, "monotonic")])
def test_bad_execution_context_clears_queue(context):
    queue, source, chunk = setup_queue()
    queue.accept(chunk, source)
    with pytest.raises(ValueError):
        queue.pop(context)
    assert len(queue) == 0


def test_no_silent_pending_replacement():
    queue, context, chunk = setup_queue()
    queue.accept(chunk, context)
    with pytest.raises(ValueError, match="Pending"):
        queue.accept(chunk, context)
    assert len(queue) == 2
