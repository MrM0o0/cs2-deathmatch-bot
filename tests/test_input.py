"""Regression tests for low-level input injection structs.

The keyboard INPUT union must be sized to its LARGEST member (MOUSEINPUT). If
it's only sized to KEYBDINPUT, sizeof(INPUT) is too small and SendInput
silently rejects every keyboard call on 64-bit Windows (returns 0) -- which
looks exactly like "the game blocks injection" but is really a struct bug.
"""

import ctypes

from src.input.keyboard import INPUT, KEYBDINPUT, MOUSEINPUT


def test_input_union_sized_to_largest_member():
    union_size = ctypes.sizeof(INPUT._INPUT_UNION)
    assert union_size >= ctypes.sizeof(MOUSEINPUT)
    assert union_size >= ctypes.sizeof(KEYBDINPUT)


def test_input_struct_matches_windows_expectation():
    # type (DWORD) + alignment padding + union(MOUSEINPUT-sized).
    # On 64-bit this is 40; on 32-bit, less. Derive it rather than hardcode.
    expected = ctypes.sizeof(ctypes.c_ulong)
    # account for 8-byte alignment of the pointer-containing union on 64-bit
    align = ctypes.alignment(INPUT._INPUT_UNION)
    if expected % align:
        expected += align - (expected % align)
    expected += ctypes.sizeof(INPUT._INPUT_UNION)
    assert ctypes.sizeof(INPUT) == expected
