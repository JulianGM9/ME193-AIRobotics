from render.table_view import MAX_RADIUS_PX, scaled_radius


def test_typical_depth_gives_a_reasonable_small_radius():
    # A previous bug multiplied scale directly by a pixel-count constant
    # (e.g. 18), giving radii in the thousands at typical depths (~1-3m)
    # given this project's focal_length - this pins the fix in place.
    scale = 700 / 1.9  # focal_length=700, depth=1.9m - a typical in-game depth
    radius = scaled_radius(0.02, scale)  # a ping-pong-ball-sized real object
    assert 1 <= radius <= 50


def test_radius_is_clamped_even_at_extreme_close_range():
    # Depth right at the near-clip boundary (0.06m) would otherwise blow
    # scale up to >11000, and a real-size object times that is still huge
    # without the clamp.
    scale = 700 / 0.06
    radius = scaled_radius(0.1, scale, min_px=4)
    assert radius == MAX_RADIUS_PX


def test_radius_never_drops_below_min_px():
    scale = 700 / 100.0  # very far away
    radius = scaled_radius(0.02, scale, min_px=2)
    assert radius == 2


def test_larger_real_size_gives_a_larger_radius_at_the_same_depth():
    scale = 700 / 2.0
    small = scaled_radius(0.02, scale)
    large = scaled_radius(0.15, scale)
    assert large > small
