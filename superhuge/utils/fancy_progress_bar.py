from lightning.pytorch.callbacks import RichProgressBar
import inspect
from typing import Any, Iterable, Union, cast
from lightning.pytorch.callbacks.progress.rich_progress import *
from rich.table import Table


class MetricsTableColumn(MetricsTextColumn):
    """A column containing table."""

    def __init__(
        self,
        trainer: "pl.Trainer",
        style: Union[str, "Style"],
        text_delimiter: str,
        metrics_format: str,
    ):
        super().__init__(
            trainer=trainer,
            style=style,
            text_delimiter=text_delimiter,
            metrics_format=metrics_format,
        )

    def render(self, task: "Task"):
        assert isinstance(self._trainer.progress_bar_callback, RichProgressBar)
        if (
            self._trainer.state.fn != "fit"
            or self._trainer.sanity_checking
            or self._trainer.progress_bar_callback.train_progress_bar_id != task.id
        ):
            return Text()
        if self._trainer.training and task.id not in self._tasks:
            self._tasks[task.id] = "None"
            if self._renderable_cache:
                self._current_task_id = cast(TaskID, self._current_task_id)
                self._tasks[self._current_task_id] = self._renderable_cache[
                    self._current_task_id
                ][1]
            self._current_task_id = task.id
        if self._trainer.training and task.id != self._current_task_id:
            return self._tasks[task.id]

        self._metrics.pop("v_num", None)
        train_metrics = {
            k.split(
                "/",
            )[-1]: v
            for k, v in self._metrics.items()
            if "train" in k
        }
        val_metrics = {
            k.split(
                "/",
            )[-1]: v
            for k, v in self._metrics.items()
            if "val" in k
        }
        test_metrics = {
            k.split(
                "/",
            )[-1]: v
            for k, v in self._metrics.items()
            if "test" in k
        }

        all_keys = sorted(
            set(
                [
                    key.split(
                        "/",
                    )[-1]
                    for key in self._metrics.keys()
                ]
            )
        )

        # Construct a table
        table = Table(
            title="Training Metrics", show_header=True, header_style="bold magenta"
        )
        table.add_column("Metric", style="bold cyan", justify="left")
        table.add_column("Train", style="green", justify="right")
        table.add_column("Validation", style="yellow", justify="right")
        table.add_column("Test", style="red", justify="right")

        for key in all_keys:
            train_val = (
                f"{train_metrics.get(key, 'N/A'):.4f}" if key in train_metrics else "—"
            )
            val_val = (
                f"{val_metrics.get(key, 'N/A'):.4f}" if key in val_metrics else "—"
            )
            test_val = (
                f"{test_metrics.get(key, 'N/A'):.4f}" if key in test_metrics else "—"
            )
            table.add_row(key, train_val, val_val, test_val)
        return table


class MetricNextLineProgress(CustomProgress):
    def __init__(self, metric_table: MetricsTableColumn, *columns, **kwargs):
        self._metric_table = metric_table
        super().__init__(*columns, **kwargs)

    def get_renderables(self) -> Iterable[RenderableType]:
        """Get a number of renderables for the progress display."""
        yield self.make_tasks_table(self.tasks)
        yield self.make_metric_tasks_table(self.tasks)

    def make_metric_tasks_table(self, tasks: Iterable[Task]) -> Table:
        table_columns = self._metric_table.get_table_column().copy()
        table = Table.grid(table_columns, padding=(0, 1), expand=self.expand)

        for task in tasks:
            if task.visible:
                table.add_row(self._metric_table(task))
        return table


class FancyProgressBar(RichProgressBar):
    def __init__(self, refresh_rate: int = 5):
        parent_signature = inspect.signature(super().__init__)

        # Validate the arguments against the parent's signature
        bound_arguments = parent_signature.bind(refresh_rate=refresh_rate)
        bound_arguments.apply_defaults()  # Ensure default values are included

        # Forward the validated arguments to the parent
        super().__init__(*bound_arguments.args, **bound_arguments.kwargs)

    def _init_progress(self, trainer: "pl.Trainer") -> None:
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            reconfigure(**self._console_kwargs)
            self._console = get_console()
            self._console.clear_live()
            self._metric_component = MetricsTableColumn(
                trainer,
                self.theme.metrics,
                self.theme.metrics_text_delimiter,
                self.theme.metrics_format,
            )
            self.progress = MetricNextLineProgress(
                self._metric_component,
                *self.configure_columns(trainer),
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,
                refresh_per_second=1 / self._refresh_rate,
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False
