# vlc_tools/vlc_serial.py
import re
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Any, List

try:
    import serial  # pyserial
except Exception as e:
    serial = None

NL = "\n"

MSG_RE = re.compile(r"^m\[(.*)\]$")
STAT_RE = re.compile(r"^s\[(.*)\]$")
VERS_RE = re.compile(r"^p\[(.*)\]$")
ADDR_RE = re.compile(r"^a\[(.*)\]$")

def _split_top(msg: str) -> List[str]:
    # Split commas; protocol fields do not contain nested brackets here (except message, handled by 'm[R,...]')
    parts = [p.strip() for p in msg.split(",")]
    return parts

def _parse_srcdest(token: str):
    if "->" in token:
        src, dest = token.split("->", 1)
        return src.strip(), dest.strip()
    return token.strip(), None

@dataclass
class VlcStat:
    mode: str
    stype: str
    src: Optional[str] = None
    dest: Optional[str] = None
    size: Optional[int] = None
    txsize: Optional[int] = None
    seq: Optional[int] = None
    cw: Optional[int] = None
    cwsize: Optional[int] = None
    dispatch_ms: Optional[float] = None
    time_ms: Optional[float] = None
    raw: str = ""

@dataclass
class VlcMessage:
    mtype: str                    # 'T' (tx done by PHY), 'D' (dispatch done), 'R' (received)
    subtype: Optional[str] = None # for 'R': 'D','A','R','C'
    payload: Optional[str] = None
    raw: str = ""

@dataclass
class VlcEvent:
    kind: str                     # 'm', 's', 'a', 'p', 'raw'
    message: Optional[VlcMessage] = None
    stat: Optional[VlcStat] = None
    text: str = ""

class VlcDevice:
    """
    High-level wrapper for ETH VLC device serial protocol.
    """
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0, autostart: bool = True):
        if serial is None:
            raise RuntimeError("pyserial is not installed. Run: pip install pyserial")
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2.0)  # give device time to boot after implicit reset
        self._handlers: List[Callable[[VlcEvent], None]] = []
        self._alive = False
        self._thread: Optional[threading.Thread] = None
        if autostart:
            self.start()

    # --- lifecycle ---
    def start(self):
        if self._alive:
            return
        self._alive = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._alive = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass

    # --- handlers ---
    def add_handler(self, fn: Callable[[VlcEvent], None]):
        self._handlers.append(fn)

    def _emit(self, ev: VlcEvent):
        for fn in list(self._handlers):
            try:
                fn(ev)
            except Exception:
                # keep going for other handlers
                pass

    # --- low level I/O ---
    def write_line(self, line: str):
        if not line.endswith(NL):
            line += NL
        self._ser.write(line.encode("utf-8", errors="ignore"))

    def reset(self):
        self.write_line("r")

    def get_version(self) -> Optional[str]:
        self.write_line("p")
        # version will be sent asynchronously as p[vX]
        return None

    def get_address(self):
        self.write_line("a")

    def set_address(self, addr_hex: str):
        self.write_line(f"a[{addr_hex}]")

    def config(self, group: int, param: int, value: int):
        self.write_line(f"c[{group},{param},{value}]")

    # Convenience config helpers (per assignment tables)
    def set_retransmissions(self, n: int):
        self.config(1, 0, int(n))

    def set_difs(self, slots: int):
        self.config(1, 1, int(slots))

    def set_cwmin(self, v: int):
        self.config(1, 2, int(v))

    def set_cwmax(self, v: int):
        self.config(1, 3, int(v))

    def set_fec_threshold(self, v: int):
        # PHY group [0], param 1
        self.config(0, 1, int(v))

    def set_channel_busy_threshold(self, v: int):
        # PHY group [0], param 2
        self.config(0, 2, int(v))

    def set_light_emission(self, enable: bool):
        self.config(0, 3, 1 if enable else 0)

    def set_logging_level(self, level: int):
        self.config(2, 0, int(level))

    def send_message(self, payload: bytes, dest_hex: str) -> None:
    # decode payload to text and escape backslashes/newlines so they survive the command format
    safe = payload.decode("utf-8", errors="ignore")
    safe = safe.replace("\\", "\\\\").replace("\n", "\\\\n")
    # ensure textual terminator '\0' (two chars) is present
    if not safe.endswith("\\0"):
        safe = safe + "\\0"
    # enforce max 200 chars payload (excl. protocol overhead)
    if len(safe) > 200:
        safe = safe[:200]
    self.write_line(f"m[{safe},{dest_hex}]")

    # --- reading & parsing ---
    def _read_loop(self):
        buf = ""
        while self._alive:
            try:
                b = self._ser.read(1)
                if not b:
                    continue
                ch = chr(b[0])
                if ch == "\n":
                    line = buf
                    buf = ""
                    if not line:
                        continue
                    self._handle_line(line)
                else:
                    buf += ch
            except Exception:
                # swallow errors; keep loop alive
                time.sleep(0.01)

    def _handle_line(self, line: str):
        line = line.strip()
        # raw taps
        if line.startswith("m["):
            m = MSG_RE.match(line)
            if m:
                body = m.group(1)
                # m[T] or m[D] or m[R,type,message]
                if body == "T":
                    ev = VlcEvent(kind="m", message=VlcMessage(mtype="T", raw=line), text=line)
                    self._emit(ev)
                    return
                if body == "D":
                    ev = VlcEvent(kind="m", message=VlcMessage(mtype="D", raw=line), text=line)
                    self._emit(ev)
                    return
                if body.startswith("R"):
                    # format R,type,message
                    # message can contain commas; split only first two commas
                    try:
                        _, rest = body.split("R", 1)
                        rest = rest.lstrip(",")
                        parts = rest.split(",", 1)
                        subtype = parts[0].strip()
                        payload = parts[1] if len(parts) > 1 else ""
                    except Exception:
                        subtype, payload = None, ""
                    ev = VlcEvent(kind="m", message=VlcMessage(mtype="R", subtype=subtype, payload=payload, raw=line), text=line)
                    self._emit(ev)
                    return
        elif line.startswith("s["):
            m = STAT_RE.match(line)
            if m:
                body = m.group(1)
                parts = _split_top(body)
                # Expected fields:
                # mode, type, src->dest, size(txsize), seq, cw, cwsize, dispatch, time
                st = VlcStat(mode=None, stype=None, raw=line)
                try:
                    st.mode = parts[0]
                    st.stype = parts[1]
                    st.src, st.dest = _parse_srcdest(parts[2])
                    # payload size; txsize may be "N(M)"
                    if "(" in parts[3] and parts[3].endswith(")"):
                        size_str, txsize_str = parts[3].split("(", 1)
                        txsize_str = txsize_str[:-1]
                        st.size = int(size_str)
                        st.txsize = int(txsize_str)
                    else:
                        st.size = int(parts[3])
                        st.txsize = None
                    st.seq = int(parts[4])
                    # cw and cwsize may be absent for Rx
                    st.cw = int(parts[5]) if len(parts) > 5 and parts[5] != "" else None
                    st.cwsize = int(parts[6]) if len(parts) > 6 and parts[6] != "" else None
                    # dispatch, time (ms)
                    disp_idx = 7
                    time_idx = 8
                    if len(parts) > disp_idx and parts[disp_idx] != "":
                        st.dispatch_ms = float(parts[disp_idx])
                    if len(parts) > time_idx and parts[time_idx] != "":
                        st.time_ms = float(parts[time_idx])
                except Exception:
                    pass
                ev = VlcEvent(kind="s", stat=st, text=line)
                self._emit(ev)
                return
        elif line.startswith("a["):
            m = ADDR_RE.match(line)
            if m:
                ev = VlcEvent(kind="a", text=line)
                self._emit(ev)
                return
        elif line.startswith("p["):
            m = VERS_RE.match(line)
            if m:
                ev = VlcEvent(kind="p", text=line)
                self._emit(ev)
                return
        # fallback
        self._emit(VlcEvent(kind="raw", text=line))

# helpers
def safe_sleep_ms(ms: int):
    time.sleep(max(0.0, ms) / 1000.0)
