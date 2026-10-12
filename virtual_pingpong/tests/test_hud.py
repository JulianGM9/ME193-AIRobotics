from render.hud import Countdown, FeedbackBanner, SwingFlash


# --- FeedbackBanner ---

def test_banner_not_visible_before_anything_is_shown():
    banner = FeedbackBanner(duration_seconds=1.0)
    assert banner.is_visible(now=0.0) is False
    assert banner.alpha(now=0.0) == 0


def test_banner_fully_visible_right_when_shown():
    banner = FeedbackBanner(duration_seconds=1.0)
    banner.show("Too early!", now=5.0)
    assert banner.alpha(now=5.0) == 255
    assert banner.is_visible(now=5.0) is True


def test_banner_fades_out_linearly():
    banner = FeedbackBanner(duration_seconds=1.0)
    banner.show("Swing!", now=0.0)
    assert banner.alpha(now=0.5) == 127  # halfway through, ~half alpha (int-truncated)


def test_banner_invisible_after_duration_elapses():
    banner = FeedbackBanner(duration_seconds=1.0)
    banner.show("Wrong spot!", now=0.0)
    assert banner.is_visible(now=1.0) is False
    assert banner.alpha(now=1.5) == 0


def test_showing_a_new_message_restarts_the_fade():
    banner = FeedbackBanner(duration_seconds=1.0)
    banner.show("Too late!", now=0.0)
    banner.show("Swing!", now=0.9)  # re-shown just before the first would expire
    assert banner.message == "Swing!"
    assert banner.alpha(now=0.9) == 255


# --- Countdown ---

def test_countdown_not_active_before_started():
    countdown = Countdown(duration_seconds=3.0)
    assert countdown.is_active(now=0.0) is False
    assert countdown.is_done(now=0.0) is False


def test_countdown_active_and_counting_down():
    countdown = Countdown(duration_seconds=3.0)
    countdown.start(now=10.0)
    assert countdown.is_active(now=11.0) is True
    assert countdown.remaining(now=11.0) == 2.0


def test_countdown_done_after_duration_elapses():
    countdown = Countdown(duration_seconds=3.0)
    countdown.start(now=0.0)
    assert countdown.is_done(now=3.0) is True
    assert countdown.is_active(now=3.0) is False


def test_countdown_remaining_never_goes_negative():
    countdown = Countdown(duration_seconds=2.0)
    countdown.start(now=0.0)
    assert countdown.remaining(now=10.0) == 0.0


# --- SwingFlash ---

def test_flash_zero_before_any_trigger():
    flash = SwingFlash(duration_seconds=0.3)
    assert flash.intensity(now=0.0) == 0.0


def test_flash_full_intensity_right_when_triggered():
    flash = SwingFlash(duration_seconds=0.3)
    flash.trigger(now=5.0)
    assert flash.intensity(now=5.0) == 1.0


def test_flash_decays_linearly():
    flash = SwingFlash(duration_seconds=1.0)
    flash.trigger(now=0.0)
    assert flash.intensity(now=0.5) == 0.5


def test_flash_zero_after_duration_elapses():
    flash = SwingFlash(duration_seconds=0.3)
    flash.trigger(now=0.0)
    assert flash.intensity(now=0.3) == 0.0
    assert flash.intensity(now=1.0) == 0.0


def test_retriggering_restarts_the_flash():
    flash = SwingFlash(duration_seconds=1.0)
    flash.trigger(now=0.0)
    flash.trigger(now=0.9)  # re-triggered just before the first would fade out
    assert flash.intensity(now=0.9) == 1.0
