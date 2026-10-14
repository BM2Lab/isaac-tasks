"""Simulator-authoritative chunk execution: physical rows, episode-bound, no interpolation."""
from collections import deque
from contracts.actions import ActionChunk


class ActionQueue:
    def __init__(self, spec):
        self.spec = spec
        self.clear()

    def clear(self):
        self.rows = deque()
        self.source = None
        self.deadline = None
        self.next_step = None

    def __len__(self):
        return len(self.rows)

    def accept(self, chunk: ActionChunk, current):
        if chunk.observation != current:
            raise ValueError("Chunk must reference the exact current capture context")
        chunk.validate_for(self.spec, current)
        if self.rows:
            raise ValueError("Pending chunk must be consumed or cleared by reset; no silent replacement")
        self.rows = deque(chunk.values)
        self.source, self.deadline, self.next_step = chunk.observation, chunk.valid_until_ns, current.step_index

    def pop(self, current):
        if not self.rows:
            raise ValueError("No queued action: simulator remains paused")
        if (current.episode_id != self.source.episode_id or current.clock != self.source.clock
                or current.step_index != self.next_step or current.timestamp_ns < self.source.timestamp_ns
                or current.timestamp_ns >= self.deadline):
            self.clear()
            raise ValueError("Queued action invalidated by episode, step, clock or expiry")
        self.next_step += 1
        return self.rows.popleft()
