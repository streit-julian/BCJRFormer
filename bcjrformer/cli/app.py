from rich.console import Console
from rich.theme import Theme
from rich.logging import RichHandler
import typer
import logging


class BCJRFormerTyper(typer.Typer):
    console: Console

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        theme = Theme(
            {
                "prompt.choices": "bold blue",
            }
        )

        self.console = Console(
            highlight=False,
            theme=theme,
            color_system="auto",
        )

    def get_logger(self, name="dna_ecct", level=logging.INFO) -> logging.Logger:
        logger = logging.getLogger(name)

        formatter = logging.Formatter("%(message)s")

        richHandler = RichHandler(console=self.console, log_time_format="[%X]")

        richHandler.setFormatter(formatter)

        logger.addHandler(richHandler)

        logger.setLevel(level)

        return logger


app = BCJRFormerTyper(name="BCJRFormer CLI", no_args_is_help=True, add_completion=True)
