"""Capture real relative mouse movement via the Win32 Raw Input API.

CS2 locks and hides the cursor in-game, so GetCursorPos / low-level mouse hooks
report a clamped position with ~zero delta -- useless for recording how the
player turns. Raw Input reports the device's raw relative motion (lLastX/Y)
regardless of cursor clamping, which is exactly the mouse-turn ground truth the
movement model needs.

A hidden message-only window receives WM_INPUT messages on a background thread;
deltas accumulate and are drained with read_delta().
"""

import ctypes
import threading
from ctypes import wintypes

WM_INPUT = 0x00FF
WM_CLOSE = 0x0010
RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0
RIDEV_INPUTSINK = 0x00000100
HWND_MESSAGE = wintypes.HWND(-3)

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(
    LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
)

user32 = ctypes.windll.user32


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class RAWMOUSE(ctypes.Structure):
    # Layout per Win32 docs; the button union is read as a single ULONG (_pad
    # aligns it). We only use lLastX / lLastY (relative motion).
    _fields_ = [
        ("usFlags", wintypes.USHORT),
        ("_pad", wintypes.USHORT),
        ("ulButtons", wintypes.ULONG),
        ("ulRawButtons", wintypes.ULONG),
        ("lLastX", wintypes.LONG),
        ("lLastY", wintypes.LONG),
        ("ulExtraInformation", wintypes.ULONG),
    ]


class RAWINPUT(ctypes.Structure):
    _fields_ = [("header", RAWINPUTHEADER), ("mouse", RAWMOUSE)]


class WNDCLASS(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HANDLE),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HANDLE),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


class RawMouseListener:
    """Background Raw Input listener accumulating relative mouse motion."""

    def __init__(self):
        self._dx = 0
        self._dy = 0
        self._lock = threading.Lock()
        self._hwnd = None
        self._thread = None
        self._wndproc_ref = None  # keep the callback alive

    def read_delta(self) -> tuple[int, int]:
        """Return motion accumulated since the last call, then reset to zero."""
        with self._lock:
            dx, dy = self._dx, self._dy
            self._dx = self._dy = 0
        return dx, dy

    def _on_input(self, lparam) -> None:
        size = wintypes.UINT(0)
        user32.GetRawInputData(
            lparam, RID_INPUT, None, ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER)
        )
        buf = ctypes.create_string_buffer(size.value)
        got = user32.GetRawInputData(
            lparam, RID_INPUT, buf, ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER)
        )
        if got != size.value:
            return
        ri = ctypes.cast(buf, ctypes.POINTER(RAWINPUT)).contents
        if ri.header.dwType == RIM_TYPEMOUSE:
            with self._lock:
                self._dx += ri.mouse.lLastX
                self._dy += ri.mouse.lLastY

    def _run(self) -> None:
        def wndproc(hwnd, msg, wparam, lparam):
            if msg == WM_INPUT:
                self._on_input(lparam)
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        self._wndproc_ref = WNDPROC(wndproc)
        hinst = ctypes.windll.kernel32.GetModuleHandleW(None)
        cls = WNDCLASS()
        cls.lpfnWndProc = self._wndproc_ref
        cls.hInstance = hinst
        cls.lpszClassName = "CS2BotRawMouse"
        user32.RegisterClassW(ctypes.byref(cls))
        self._hwnd = user32.CreateWindowExW(
            0, "CS2BotRawMouse", "rawmouse", 0, 0, 0, 0, 0, HWND_MESSAGE, None, hinst, None
        )

        rid = RAWINPUTDEVICE(0x01, 0x02, RIDEV_INPUTSINK, self._hwnd)  # generic mouse
        if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
            raise OSError("RegisterRawInputDevices failed")

        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._hwnd:
            user32.PostMessageW(self._hwnd, WM_CLOSE, 0, 0)
