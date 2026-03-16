# ABOUTME: Tests for the sweetspot_table script that generates SFT→DPO analysis tables.
# ABOUTME: Covers URL parsing, run matching, metric extraction, and table formatting.

import pytest

from scripts.sweetspot_table import parse_wandb_url


class TestParseWandbUrl:
    def test_parses_full_url(self):
        url = "https://wandb.ai/shougan-university-of-waterloo/gsm8k-llama3-3B"
        assert parse_wandb_url(url) == ("shougan-university-of-waterloo", "gsm8k-llama3-3B")

    def test_parses_url_with_trailing_slash(self):
        url = "https://wandb.ai/shougan-university-of-waterloo/gsm8k-llama3-3B/"
        assert parse_wandb_url(url) == ("shougan-university-of-waterloo", "gsm8k-llama3-3B")

    def test_parses_url_with_query_params(self):
        url = "https://wandb.ai/shougan-university-of-waterloo/gsm8k-llama3-3B?nw=nwusershougan"
        assert parse_wandb_url(url) == ("shougan-university-of-waterloo", "gsm8k-llama3-3B")

    def test_rejects_invalid_url(self):
        with pytest.raises(ValueError):
            parse_wandb_url("not-a-url")

    def test_rejects_non_wandb_url(self):
        with pytest.raises(ValueError):
            parse_wandb_url("https://example.com/foo/bar")
