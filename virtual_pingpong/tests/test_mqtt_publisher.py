from net.mqtt_publisher import format_payload


def test_integer_streak_formats_with_one_decimal():
    assert format_payload(7) == "7.0"


def test_zero_formats_correctly():
    assert format_payload(0) == "0.0"


def test_already_float_streak_formats_correctly():
    assert format_payload(12.0) == "12.0"
