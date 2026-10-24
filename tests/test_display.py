import pytest

from jetkvm_mcp.display import HID_ABS_MAX, ScreenGeometry


def test_native_size_kept_when_it_fits():
    geo = ScreenGeometry(1920, 1080, max_edge=1920)
    assert (geo.width, geo.height, geo.scale) == (1920, 1080, 1.0)


def test_downscales_to_max_edge():
    geo = ScreenGeometry(3840, 2160, max_edge=1920)
    assert (geo.width, geo.height) == (1920, 1080)
    geo = ScreenGeometry(1080, 1920, max_edge=1280)  # portrait
    assert (geo.width, geo.height) == (720, 1280)


def test_to_hid_targets_pixel_centers_within_range():
    geo = ScreenGeometry(1920, 1080, max_edge=1920)
    assert geo.to_hid(0, 0) == (9, 15)
    x, y = geo.to_hid(1919, 1079)
    assert HID_ABS_MAX - 10 < x <= HID_ABS_MAX and HID_ABS_MAX - 16 < y <= HID_ABS_MAX
    # Mapping back the way a host does (abs / 32767 * width) lands on the same pixel.
    for px in (0, 1, 500, 959, 1918, 1919):
        hx, _ = geo.to_hid(px, 0)
        assert int(hx / HID_ABS_MAX * 1920) == px


def test_to_hid_is_resolution_independent():
    # The center of a downscaled 4K screenshot is the center of the host screen.
    x, y = ScreenGeometry(3840, 2160, max_edge=1280).to_hid(640, 360)
    assert abs(x - HID_ABS_MAX / 2) < 30 and abs(y - HID_ABS_MAX / 2) < 30


@pytest.mark.parametrize("x, y", [(-1, 0), (0, -1), (1920, 0), (0, 1080)])
def test_to_hid_rejects_out_of_range(x, y):
    with pytest.raises(ValueError, match="outside the screen"):
        ScreenGeometry(1920, 1080, max_edge=1920).to_hid(x, y)


def test_native_box_scales_up():
    geo = ScreenGeometry(3840, 2160, max_edge=1920)
    assert geo.to_native_box(100, 100, 50, 25) == (200, 200, 300, 250)
    # Clamped to the screen.
    assert geo.to_native_box(1900, 1070, 100, 100) == (3800, 2140, 3840, 2160)
