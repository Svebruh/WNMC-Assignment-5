# chat_cli.py
import argparse
import threading
import sys
import time
from vlc_tools.vlc_serial import VlcDevice, safe_sleep_ms

def set_address_with_retry(dev: VlcDevice, addr: str):
    # Apply best-practice from assignment notes
    dev.reset()
    time.sleep(0.5)
    dev.set_address(addr)
    time.sleep(5.0)
    dev.set_address(addr)
    time.sleep(0.2)
    dev.get_address()

def on_event(ev):
    if ev.kind == "m" and ev.message:
        if ev.message.mtype == "R" and ev.message.subtype == "D":
            # data message received
            print(f"\n[RECEIVED] {ev.message.payload}")
            print("> ", end="", flush=True)
    elif ev.kind == "a":
        print(f"[Device address echo] {ev.text}")
    elif ev.kind == "p":
        print(f"[Version] {ev.text}")

def reader(dev: VlcDevice):
    dev.add_handler(on_event)
    while True:
        time.sleep(0.2)

def main():
    ap = argparse.ArgumentParser(description="Simple VLC chat (broadcast by default).")
    ap.add_argument("--port", required=True, help="Serial port, e.g., COM4 or /dev/ttyACM0")
    ap.add_argument("--address", required=True, help="Local device address in hex, e.g., AB")
    ap.add_argument("--dest", default="FF", help="Destination address in hex (default: FF broadcast)")
    ap.add_argument("--nickname", default="", help="Optional nickname prefix")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    dev = VlcDevice(args.port, baudrate=args.baud)
    set_address_with_retry(dev, args.address)
    dev.set_logging_level(0)

    print("VLC Chat started. Type messages and press Enter. Use /quit to exit.")
    print(f"Local address: {args.address}  Dest: {args.dest}")
    threading.Thread(target=reader, args=(dev,), daemon=True).start()

    try:
        while True:
            msg = input("> ")
            if msg.strip().lower() == "/quit":
                break
            if args.nickname:
                msg = f"[{args.nickname}] {msg}"
            dev.send_message(msg.encode("utf-8"), args.dest)
            safe_sleep_ms(20)  # be gentle to serial interface
    except KeyboardInterrupt:
        pass
    finally:
        dev.stop()

if __name__ == "__main__":
    main()
