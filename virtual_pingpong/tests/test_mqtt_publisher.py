from net.mqtt_publisher import format_payload


def test_format_payload_includes_all_three_values_readably():
    message = format_payload(current_streak=3, record_streak=7, player_score=4, opponent_score=2)
    assert "3" in message
    assert "7" in message
    assert "4-2" in message


def test_format_payload_is_plain_text_not_json():
    message = format_payload(current_streak=0, record_streak=0, player_score=0, opponent_score=0)
    assert not message.strip().startswith("{")


def test_format_payload_labels_current_and_best_streak_distinctly():
    message = format_payload(current_streak=3, record_streak=7, player_score=0, opponent_score=0)
    assert "Current streak: 3" in message
    assert "Best streak: 7" in message


def test_format_payload_shows_score_as_player_dash_opponent():
    message = format_payload(current_streak=0, record_streak=0, player_score=9, opponent_score=5)
    assert "Score: 9-5" in message
