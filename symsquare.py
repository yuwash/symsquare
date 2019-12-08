# -*- coding: utf-8 -*-
from __future__ import print_function
from argparse import ArgumentParser
from functools import partial


def ascii_map_x(value):
    return 'X' if value else '-'


def ascii_map_aske(value):
    return 'g' if value else 'f'


def draw_ascii(bitmap, out=None, map_function=ascii_map_x):
    """
    >>> draw_ascii([(1, 0), (1, 0, 0)])
    X-
    X--
    >>> draw_ascii([int_to_bits(5, 3), int_to_bits(6, 3)])
    X-X
    -XX
    """
    print_kwargs = {} if out is None else {"file": out}
    for row in bitmap:
        print(''.join(map(map_function, row)), **print_kwargs)


def abs_to_ref(x, y, size):
    """
    >>> abs_to_ref(1, 1, 3)
    (0, 0)
    >>> abs_to_ref(1, 2, 4), abs_to_ref(2, 2, 5), abs_to_ref(2, 3, 6)
    ((0, 0), (0, 0), (0, 0))
    >>> abs_to_ref(0, 2, 3)
    (1, 1)
    >>> abs_to_ref(0, 3, 4), abs_to_ref(1, 3, 5), abs_to_ref(1, 4, 6)
    ((1, 1), (1, 1), (1, 1))
    >>> abs_to_ref(2, 2, 3), abs_to_ref(2, 0, 3), abs_to_ref(0, 2, 3)
    ((1, 1), (1, 1), (1, 1))
    >>> abs_to_ref(2, 0, 4)
    (0, 1)
    >>> abs_to_ref(0, 1, 4), abs_to_ref(1, 3, 4), abs_to_ref(3, 2, 4)
    ((0, 1), (0, 1), (0, 1))
    """
    if size % 2:
        ref_2x, ref_2y = abs_to_ref(2*x, 2*y, 2*size)
        # everything apart from the axis is moved up right
        if 1 <= ref_2x:
            ref_2x += 1
        if 1 <= ref_2y:
            ref_2y += 1
        return ref_2x//2, ref_2y//2
    # work in double the resolution and add half (1) for center of pixel
    ref_x_2 = 2*x + 1 - size  # move axis
    ref_y_2 = size - 2*y - 1  # ref coordinates go from bottom to top
    for _ in range(get_quadrant(ref_x_2, ref_y_2)):
        ref_x_2, ref_y_2 = ref_y_2, -ref_x_2
        # rotation 90 deg clockwise
    return ref_x_2//2, ref_y_2//2


def get_quadrant(x, y):
    """
    >>> from itertools import product
    >>> [get_quadrant(x, y) for x, y in product([-1, 1], [-1, 1])]
    [2, 1, 3, 0]
    >>> get_quadrant(-1, 0)
    1
    """
    return 2*(y < 0) + 1*((x < 0) ^ (y < 0))


def tridata_to_bitmap(tridata, size):
    for row in range(size):
        yield tridata_to_bitmap_row(tridata, size, row)


def triangle_volume(h):
    return h*(h + 1)//2  # Gaussian sum


def get_from_triangle(x, y, triangle):
    """
    >>> get_from_triangle(0, 0, [1])
    1
    >>> there = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)]
    >>> triangle = list(range(6))
    >>> [get_from_triangle(x, y, triangle) for x, y in there]
    [0, 1, 2, 3, 4, 5]
    """
    if x > y:
        raise IndexError('x can not exceed y={}, {} given'.format(y, x))
    complete_rows_volume = triangle_volume(y)
    return triangle[complete_rows_volume + x]


def get_from_triangle_upsidedown(x, y, triangle, length):
    """
    >>> get_from_triangle_upsidedown(0, 0, [1], 1)
    1
    >>> there = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)]
    >>> triangle = list(range(6))
    >>> [get_from_triangle_upsidedown(x, y, triangle, 3) for x, y in there]
    [0, 1, 2, 3, 4, 5]
    """
    y_from_bottom = length - y
    if x > y_from_bottom:
        raise IndexError(
            'x can not exceed y={}, {} given'.format(y_from_bottom, x))
    complete_rows_volume = len(triangle) - triangle_volume(y_from_bottom)
    return triangle[complete_rows_volume + x]


def tridata_to_bitmap_row(tridata, size, row):
    """
    >>> tridata = [3], [2, 1], [0]
    >>> [tuple(tridata_to_bitmap_row(tridata, 4, i)) for i in range(4)]
    [(1, 3, 0, 1), (0, 2, 2, 3), (3, 2, 2, 0), (1, 0, 3, 1)]
    >>> tridata = [1], [1, 0], []
    >>> draw_ascii(tridata_to_bitmap(tridata, 3))
    -X-
    XXX
    -X-
    >>> tridata = int_to_bits(1, 3), int_to_bits(1, 3), [1]
    >>> draw_ascii(tridata_to_bitmap(tridata, 5))
    ---X-
    X-X--
    -XXX-
    --X-X
    -X---
    """
    lower, diag, upper = tridata
    # tridata goes from bottom to top, left to right
    reflen = sum(divmod(size, 2))  # ceil
    for col in range(size):
        ref_x, ref_y = abs_to_ref(col, row, size)
        if size % 2 and ref_x == 0:
            ref_x, ref_y = ref_y, 0
            # use the lower triangle for the central column/row
        if ref_y < ref_x:
            yield get_from_triangle_upsidedown(
                ref_x - ref_y - 1, ref_y, lower, reflen - 1)
            # begin x at the column next to the diagonal
        elif ref_x == ref_y:
            yield diag[ref_x]
        else:
            center_shift = size % 2
            # upper doesn’t contain first column/row if central
            yield get_from_triangle(
                ref_x - center_shift, ref_y - center_shift - 1, upper)
            # always shift y for diagonal


def build_variants(size):
    reflen = sum(divmod(size, 2))  # ceil
    lower_vol = triangle_volume(reflen - 1)
    upper_vol = lower_vol - (reflen - 1)*(size % 2)
    # remove central column from upper (only relevant for odd size)
    for l_value in range(2**lower_vol):
        lower = int_to_bits(l_value, lower_vol)
        for u_value in range(l_value % (2**upper_vol) + 1):
            # eliminate symmetric, therefore only u_value <= l_value
            upper = int_to_bits(u_value, upper_vol)
            for d_value in range(2**(reflen - 1)):
                # the outermost pixel is always 0 to remove half of the
                # variations because the inverse should exist
                diag = int_to_bits(d_value, reflen)
                yield lower, diag, upper


def int_to_bits(value, length):
    """
    >>> int_to_bits(1, 3)
    (1, 0, 0)
    >>> int_to_bits(7, 2)
    (1, 1)
    """
    return tuple((value >> i) % 2 for i in range(length))


def classic_output(bitmaps, out=None):
    print_kwargs = {} if out is None else {"file": out}
    for i, bitmap in enumerate(bitmaps):
        print(i, **print_kwargs)
        draw_ascii(bitmap, out=out)


def aske_output(bitmaps, out=None):
    print_kwargs = {} if out is None else {"file": out}
    print(
        '---\n'
        '# generated with symsquare\n'
        'background-color: 0xffffffff\n'  # same color used in ascii_map_aske
        '...', **print_kwargs)
    for bitmap in bitmaps:
        print('', **print_kwargs)  # just a blank line
        draw_ascii(bitmap, out=out, map_function=ascii_map_aske)


if __name__ == "__main__":
    argparser = ArgumentParser(description='Draw symmetric square variations')
    argparser.add_argument(
        'size', type=int, default=3, nargs='?', help='canvas size')
    argparser.add_argument(
        '--aske', action='store_true',
        help='output in aske format (yuwash/askiisketch)')
    parsed_args = argparser.parse_args()
    size = parsed_args.size
    variants_bitmap = map(
        partial(tridata_to_bitmap, size=size), build_variants(size))
    if parsed_args.aske:
        aske_output(variants_bitmap)
    else:
        classic_output(variants_bitmap)
