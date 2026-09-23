"""Backward-compatible training CLI. Prefer: python -m app.ml.house_price.train"""

from app.ml.house_price.train import main

if __name__ == "__main__":
    main()
