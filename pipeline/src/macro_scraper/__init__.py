"""Scraping et normalisation des indicateurs macro hebdomadaires."""

from __future__ import annotations

from src.macro_scraper.pipeline import run_scrape_macro, scrape_macro_dataframe

__all__ = ["run_scrape_macro", "scrape_macro_dataframe"]
