from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

_progress_instance = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeRemainingColumn(),
    transient=True,
)


def get_progress():
    """get progress instance"""
    _progress_instance.start()
    return _progress_instance
