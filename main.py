import argparse

from ingestion.ingest_data import ingest
from chatbot.chat_interface import run_cli


def main():
    parser = argparse.ArgumentParser(description="RAG Bot")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest documents from data/documents before starting",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Streamlit web interface",
    )
    args = parser.parse_args()

    if args.ingest:
        chunks = ingest()
        print(f"Ingested {chunks} chunks.")

    if args.ui:
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app.py"],
            check=True,
        )
        return

    run_cli()


if __name__ == "__main__":
    main()
