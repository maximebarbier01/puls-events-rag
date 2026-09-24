"""Unit tests for app.data.fetch — mocked HTTP calls, no real network access."""
from unittest.mock import patch

import pandas as pd
import pytest

from app.data.fetch import _flatten_coordinates, fetch_events, save_events


def _record(uid: str, lat: float | None = 49.1, lon: float | None = 6.2) -> dict:
    return {
        "uid": uid,
        "title_fr": f"Événement {uid}",
        "location_region": "Grand Est",
        "location_coordinates": {"lat": lat, "lon": lon} if lat is not None else None,
    }


def _response(results: list[dict], total_count: int):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": results, "total_count": total_count}

    return FakeResponse()


def test_fetch_events_paginates_until_total_count_reached():
    page_1 = [_record(str(i)) for i in range(100)]
    page_2 = [_record(str(i)) for i in range(100, 130)]

    with patch("app.data.fetch.requests.get") as mock_get, patch("app.data.fetch.time.sleep"):
        mock_get.side_effect = [_response(page_1, 130), _response(page_2, 130)]

        df = fetch_events("Grand Est")

    assert len(df) == 130
    assert mock_get.call_count == 2
    # location_coordinates is flattened away, not left as a raw dict column
    assert "location_coordinates" not in df.columns
    assert "location_lat" in df.columns and "location_lon" in df.columns


def test_fetch_events_filters_on_region_by_default_and_on_date():
    with patch("app.data.fetch.requests.get") as mock_get, patch("app.data.fetch.time.sleep"):
        mock_get.return_value = _response([_record("1")], 1)

        fetch_events("Grand Est", min_last_date="2026-09-25")

    where_clause = mock_get.call_args.kwargs["params"]["where"]
    assert 'location_region="Grand Est"' in where_clause
    assert "lastdate_end >= date'2026-09-25'" in where_clause


def test_fetch_events_can_filter_on_another_zone_field():
    with patch("app.data.fetch.requests.get") as mock_get, patch("app.data.fetch.time.sleep"):
        mock_get.return_value = _response([_record("1")], 1)

        fetch_events("Moselle", zone_field="location_department")

    where_clause = mock_get.call_args.kwargs["params"]["where"]
    assert where_clause == 'location_department="Moselle"'


def test_fetch_events_stops_on_empty_page_even_if_total_count_says_otherwise():
    with patch("app.data.fetch.requests.get") as mock_get, patch("app.data.fetch.time.sleep"):
        mock_get.side_effect = [_response([_record("1")], 999), _response([], 999)]

        df = fetch_events("Grand Est")

    assert len(df) == 1
    assert mock_get.call_count == 2


def test_flatten_coordinates_extracts_lat_lon_and_handles_missing():
    df = pd.DataFrame(
        [
            {"uid": "1", "location_coordinates": {"lat": 49.11, "lon": 6.18}},
            {"uid": "2", "location_coordinates": None},
        ]
    )

    result = _flatten_coordinates(df)

    assert "location_coordinates" not in result.columns
    assert result.loc[0, "location_lat"] == 49.11
    assert result.loc[0, "location_lon"] == 6.18
    assert pd.isna(result.loc[1, "location_lat"])


def test_flatten_coordinates_without_column_adds_null_lat_lon():
    df = pd.DataFrame([{"uid": "1"}])

    result = _flatten_coordinates(df)

    assert result.loc[0, "location_lat"] is None
    assert result.loc[0, "location_lon"] is None


def test_save_events_writes_a_readable_parquet_file(tmp_path):
    df = pd.DataFrame([_record("1"), _record("2", lat=None, lon=None)])
    df = _flatten_coordinates(df)
    output_path = tmp_path / "raw" / "events.parquet"

    save_events(df, output_path)

    assert output_path.exists()
    reloaded = pd.read_parquet(output_path)
    assert len(reloaded) == 2
    assert set(reloaded["uid"]) == {"1", "2"}
