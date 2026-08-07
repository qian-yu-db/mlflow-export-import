"""
Test the timestamp conversion utilities.
"""

import os
import time
import pytest

from mlflow_export_import.common.timestamp_utils import (
    utc_str_to_seconds,
    utc_str_to_millis,
    fmt_ts_millis
)


pytestmark = pytest.mark.skipif(
    not hasattr(time, "tzset"),
    reason="time.tzset() is only available on Unix"
)


_winter_seconds = 1704067200 # 2024-01-01 00:00:00 UTC
_summer_seconds = 1719792000 # 2024-07-01 00:00:00 UTC
_timezones = [ "UTC", "America/Los_Angeles", "Asia/Kolkata" ]


@pytest.fixture
def set_timezone():
    """ Set the process timezone for one test and restore it afterwards. """
    # not monkeypatch.setenv, since tzset must run after TZ is restored
    saved = os.environ.get("TZ")
    def _set(name):
        os.environ["TZ"] = name
        time.tzset()
        if name != "UTC" and time.strftime("%z", time.localtime(_winter_seconds)) == "+0000":
            pytest.skip(f"no timezone database entry for {name}")
    yield _set
    if saved is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = saved
    time.tzset()


# == Test strings without an offset are read as UTC

@pytest.mark.parametrize("timezone_name", _timezones)
def test_naive_string_is_utc(set_timezone, timezone_name):
    set_timezone(timezone_name)
    assert _winter_seconds == utc_str_to_seconds("2024-01-01 00:00:00")


@pytest.mark.parametrize("timezone_name", _timezones)
def test_naive_date_only_string_is_utc(set_timezone, timezone_name):
    set_timezone(timezone_name)
    assert _winter_seconds * 1000 == utc_str_to_millis("2024-01-01")


@pytest.mark.parametrize("timezone_name", _timezones)
def test_naive_string_is_utc_during_daylight_saving(set_timezone, timezone_name):
    set_timezone(timezone_name)
    assert _summer_seconds == utc_str_to_seconds("2024-07-01 00:00:00")


@pytest.mark.parametrize("timezone_name", _timezones)
def test_round_trip_with_fmt_ts_millis(set_timezone, timezone_name):
    set_timezone(timezone_name)
    millis = utc_str_to_millis("2024-01-01 00:00:00")
    assert "2024-01-01 00:00:00" == fmt_ts_millis(millis)


# == Test strings with an offset

@pytest.mark.parametrize("timezone_name", _timezones)
def test_utc_offset_is_honored(set_timezone, timezone_name):
    set_timezone(timezone_name)
    assert _winter_seconds == utc_str_to_seconds("2024-01-01T00:00:00+00:00")


@pytest.mark.parametrize("timezone_name", _timezones)
def test_non_utc_offset_is_honored(set_timezone, timezone_name):
    set_timezone(timezone_name)
    assert _winter_seconds - (5*3600 + 30*60) == utc_str_to_seconds("2024-01-01T00:00:00+05:30")
    assert _winter_seconds + 8*3600 == utc_str_to_seconds("2024-01-01T00:00:00-08:00")


# == Test milliseconds conversion

def test_utc_str_to_millis_returns_int():
    assert isinstance(utc_str_to_millis("2024-01-01"), int)
