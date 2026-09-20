from nurb import *


@part
def offset_bracket(
    plate_width=46.0,
    plate_depth=28.0,
    base_thickness=5.0,
    wall_height=15.0,
    vertical_hole_radius=3.0,
    side_hole_radius=2.0,
    locator_notch_depth=2.0,
):
    """An asymmetric reconstruction fixture whose source frame is offset from zero."""
    origin_x = 12.0
    origin_y = -7.0

    base = Pos(origin_x, origin_y, 0.0) * Box(
        plate_width,
        plate_depth,
        base_thickness,
        align=(Align.MIN, Align.MIN, Align.MIN),
    )
    wall = Pos(origin_x, origin_y, 0.0) * Box(
        6.0,
        plate_depth,
        wall_height,
        align=(Align.MIN, Align.MIN, Align.MIN),
    )

    first_hole = Pos(origin_x + 17.0, origin_y + 8.0, -1.0) * Cylinder(
        vertical_hole_radius,
        base_thickness + 2.0,
    )
    second_hole = Pos(origin_x + 35.0, origin_y + 20.0, -1.0) * Cylinder(
        vertical_hole_radius,
        base_thickness + 2.0,
    )
    side_hole = Pos(origin_x - 1.0, origin_y + 19.0, 10.0) * Rot(0.0, 90.0, 0.0) * Cylinder(
        side_hole_radius,
        8.0,
    )

    body = base + wall - first_hole - second_hole - side_hole
    if locator_notch_depth > 0.0:
        locator_notch = Pos(
            origin_x + plate_width - locator_notch_depth,
            origin_y + 3.0,
            -1.0,
        ) * Box(
            locator_notch_depth + 1.0,
            4.0,
            base_thickness + 2.0,
            align=(Align.MIN, Align.MIN, Align.MIN),
        )
        body -= locator_notch
    return body
