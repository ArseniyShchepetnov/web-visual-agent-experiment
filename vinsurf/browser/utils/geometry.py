"""Geometry helper class."""

import itertools
from collections.abc import Generator
from typing import Any

import numpy as np

AnyType = Any
VectorType = tuple | list | np.ndarray


class Rectangle:
    """Rectangle corners."""

    vector_size = 2

    def __init__(self, top: int, left: int, bottom: int, right: int):
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right

    def _copy_new_coords(
        self, top: int, left: int, bottom: int, right: int
    ) -> "Rectangle":
        """Copy data."""
        return Rectangle(top=top, left=left, bottom=bottom, right=right)

    @property
    def height(self) -> int:
        """Get height of the rectangle."""
        return self.bottom - self.top

    @property
    def width(self) -> int:
        """Get width of the rectangle."""
        return self.right - self.left

    @property
    def area(self) -> int:
        """Get area of the rectangle."""
        return self.width * self.height

    @property
    def perimeter(self) -> int:
        """Get perimeter of the rectangle."""
        return 2 * self.width + 2 * self.height

    def __repr__(self) -> str:
        """Return a compact debug representation."""
        return (
            f"Rectangle(t={self.top},l={self.left},"
            f"r={self.right},b={self.bottom})"
        )

    @classmethod
    def empty(cls) -> "Rectangle":
        """Construct empty."""
        return cls(0, 0, 0, 0)

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return PIL compatible tuple."""
        return self.left, self.top, self.right, self.bottom

    @property
    def coords(self) -> list[tuple[int, int]]:
        """Return PIL compatible tuple."""
        return [(self.left, self.top), (self.right, self.bottom)]

    def is_degenerate(self) -> bool:
        """Return whether top is below bottom or left is past right."""
        return (self.top >= self.bottom) or (self.left >= self.right)

    def is_empty(self) -> bool:
        """Return whether the rectangle has zero width or height."""
        return (self.top == self.bottom) or (self.left == self.right)

    def is_subset(self, other: "Rectangle") -> bool:
        """Check if this rectangle is subset of another."""
        return (
            self.top >= other.top
            and self.bottom <= other.bottom
            and self.left >= other.left
            and self.right <= other.right
        )

    def intersection_ratio(self, other: "Rectangle") -> float | None:
        """Calculate ration intersection / this."""
        intersect = self & other
        if intersect.is_degenerate():
            return None
        return self.area / intersect.area

    def is_intersect(self, other: "Rectangle") -> bool:
        """Check if this rectangle is subset of another."""
        return not (self & other).is_degenerate()

    def __eq__(self, other: AnyType) -> bool:
        """Check is equal."""
        if not isinstance(other, Rectangle):
            return NotImplemented
        return (
            (self.top == other.top)
            and (self.left == other.left)
            and (self.bottom == other.bottom)
            and (self.right == other.right)
        )

    def __hash__(self) -> int:
        """Hash rectangle coordinates."""
        return hash((self.top, self.left, self.bottom, self.right))

    def __ne__(self, other: AnyType) -> bool:
        """Check whether rectangle coordinates differ."""
        if not isinstance(other, Rectangle):
            return NotImplemented
        return (
            (self.top != other.top)
            or (self.left != other.left)
            or (self.bottom != other.bottom)
            or (self.right != other.right)
        )

    def __and__(self, other: "Rectangle") -> "Rectangle":
        """Get rectangle intersection."""
        rect = self._copy_new_coords(
            top=max(self.top, other.top),
            bottom=min(self.bottom, other.bottom),
            left=max(self.left, other.left),
            right=min(self.right, other.right),
        )
        if rect.is_degenerate():
            return rect.empty()
        return rect

    def __or__(self, other: "Rectangle") -> "Rectangle":
        """Get covering rectangle."""
        return Rectangle(
            top=min(self.top, other.top),
            bottom=max(self.bottom, other.bottom),
            left=min(self.left, other.left),
            right=max(self.right, other.right),
        )

    def __add__(self, other: AnyType) -> "Rectangle":
        """Add to the rectangle."""
        if isinstance(other, VectorType):
            if len(other) == self.vector_size:
                x_dir = other[0]
                y_dir = other[1]
                return self.__class__(
                    top=self.top + y_dir,
                    bottom=self.bottom + y_dir,
                    left=self.left + x_dir,
                    right=self.right + x_dir,
                )

            return NotImplemented

        return NotImplemented

    def __mul__(self, value: int) -> "Rectangle":
        """Add to the rectangle."""
        return self.__class__(
            top=self.top * value,
            bottom=self.bottom * value,
            left=self.left * value,
            right=self.right * value,
        )

    __rmul__ = __mul__

    def __sub__(self, other: AnyType) -> "Rectangle":
        """Add to the rectangle."""
        if isinstance(other, VectorType):
            if len(other) == self.vector_size:
                x_dir = other[0]
                y_dir = other[1]
                return self._copy_new_coords(
                    top=self.top - y_dir,
                    bottom=self.bottom - y_dir,
                    left=self.left - x_dir,
                    right=self.right - x_dir,
                )

            return NotImplemented

        return NotImplemented

    def __lshift__(self, other: AnyType) -> "Rectangle":
        """Add to bottom and right."""
        if isinstance(other, VectorType):
            if len(other) == self.vector_size:
                x_dir = other[0]
                y_dir = other[1]
                return self._copy_new_coords(
                    top=self.top,
                    bottom=self.bottom + y_dir,
                    left=self.left,
                    right=self.right + x_dir,
                )

            return NotImplemented

        return NotImplemented

    def tiles(
        self, vnum: int, hnum: int
    ) -> Generator["Rectangle", None, None]:
        """Generate tiles on the rectangle."""
        height = self.height // vnum
        width = self.width // vnum
        for i_v, i_h in itertools.product(range(vnum), range(hnum)):
            top = height * i_v
            left = width * i_h
            yield self.__class__(
                top=top, bottom=top + height, left=left, right=left + width
            )
