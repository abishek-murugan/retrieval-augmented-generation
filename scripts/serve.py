"""Start the FastAPI service locally.

    uv run python scripts/serve.py [--port 8000] [--reload]
"""

import click
import uvicorn


@click.command()
@click.option("--port", default=8000, type=int)
@click.option("--host", default="0.0.0.0")
@click.option("--reload", is_flag=True)
def main(port: int, host: str, reload: bool):
    uvicorn.run(
        "defence_rag.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=4 if not reload else 1,
    )


if __name__ == "__main__":
    main()