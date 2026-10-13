"""Snapshot persistence only; no training, environment or RNG operations."""
import json


def last_valid_metrics_step(path):
    last = None
    if not path.exists():
        return last
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
                value = record.get("sampled_steps") if isinstance(record, dict) else None
                if type(value) is int and value >= 0:
                    last = value
            except (ValueError, TypeError):
                continue
    return last


class TrainingMetricsSnapshots:
    """Independent threshold counter shared by both runner persistence paths."""
    def initialize_training_metrics_logging(self):
        if self.console_interval <= 0:
            raise ValueError("training metrics interval must be positive")
        self.next_training_metrics_log = self.console_interval
        self.last_training_metrics_step = None
        self.last_step_result = None
        self.last_completed_rows = None

    def restore_training_metrics_logging(self):
        self.next_training_metrics_log = (
            self.trainer.sampled_steps // self.console_interval + 1
        ) * self.console_interval
        self.last_training_metrics_step = last_valid_metrics_step(
            self.output_dir / "training_metrics.jsonl"
        )
        self.last_step_result = None
        self.last_completed_rows = None

    def record_training_metrics_transition(self, result, rows, writer):
        self.last_step_result, self.last_completed_rows = result, rows
        if self.trainer.sampled_steps >= self.next_training_metrics_log:
            self._emit_training_metrics_snapshot(writer)
            while self.next_training_metrics_log <= self.trainer.sampled_steps:
                self.next_training_metrics_log += self.console_interval

    def _emit_training_metrics_snapshot(self, writer):
        steps = self.trainer.sampled_steps
        if self.last_training_metrics_step == steps:
            return
        if self.last_training_metrics_step is not None and steps < self.last_training_metrics_step:
            raise RuntimeError("training metrics contain future rows; resume rollback is required")
        if self.last_step_result is None:
            raise RuntimeError("no real transition available for final training metrics snapshot")
        writer(self.last_step_result, self.last_completed_rows)
        self.last_training_metrics_step = steps

    def finalize_training_metrics_logging(self, writer):
        if self.last_training_metrics_step != self.trainer.sampled_steps:
            self._emit_training_metrics_snapshot(writer)
