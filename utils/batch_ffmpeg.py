"""Batch extraction CLI: ``python -m utils.batch_ffmpeg videos.txt``."""

from core.frame_skip import BatchFFmpeg, main

__all__ = ["BatchFFmpeg", "main"]


if __name__ == "__main__":
    main()
