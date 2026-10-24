"""Screen geometry: the coordinate space the agent sees vs. the HID digitizer range."""

from __future__ import annotations

from dataclasses import dataclass

# The JetKVM absolute pointer is a HID digitizer with logical range 0..32767.
HID_ABS_MAX = 32767


@dataclass(frozen=True)
class ScreenGeometry:
    """Maps between screenshot pixels (possibly downscaled) and the host screen.

    Screenshots are the native capture scaled so the long edge is at most
    `max_edge`. Every mouse tool takes coordinates in that screenshot space, so
    what the agent sees is exactly what it clicks.
    """

    native_width: int
    native_height: int
    max_edge: int

    def __post_init__(self) -> None:
        if self.native_width <= 0 or self.native_height <= 0:
            raise ValueError("screen size must be positive")

    @property
    def scale(self) -> float:
        return min(1.0, self.max_edge / max(self.native_width, self.native_height))

    @property
    def width(self) -> int:
        return max(1, round(self.native_width * self.scale))

    @property
    def height(self) -> int:
        return max(1, round(self.native_height * self.scale))

    def check(self, x: float, y: float) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(
                f"coordinate ({x}, {y}) is outside the screen; valid range is "
                f"x 0..{self.width - 1}, y 0..{self.height - 1}"
            )

    def to_hid(self, x: float, y: float) -> tuple[int, int]:
        """Screenshot pixel -> absolute HID coordinates, aimed at the pixel's center."""
        self.check(x, y)
        hx = round((x + 0.5) / self.width * HID_ABS_MAX)
        hy = round((y + 0.5) / self.height * HID_ABS_MAX)
        return min(max(hx, 0), HID_ABS_MAX), min(max(hy, 0), HID_ABS_MAX)

    def to_native_box(self, x: float, y: float, w: float, h: float) -> tuple[int, int, int, int]:
        """Screenshot-space rectangle -> clamped native-pixel box (left, top, right, bottom)."""
        if w <= 0 or h <= 0:
            raise ValueError("width and height must be positive")
        inv = 1 / self.scale
        left = max(0, min(self.native_width - 1, int(x * inv)))
        top = max(0, min(self.native_height - 1, int(y * inv)))
        right = max(left + 1, min(self.native_width, round((x + w) * inv)))
        bottom = max(top + 1, min(self.native_height, round((y + h) * inv)))
        return left, top, right, bottom
