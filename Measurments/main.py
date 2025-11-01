# main.py
from sys import path
path.append(".")

from utils import *
from serial_setup import *
from measurer import Measurer
from time import sleep

# --- CONFIG: payload sizes to test sequentially ---
PAYLOAD_SIZES = [1, 100, 180]
# ---------------------------------------------------

DEFAULT_PORT = default_port()
port = prompt_with_default("Use this serial port", DEFAULT_PORT)

is_measuring = prompt_with_default("This station sends payloads and does the measuring", "yes").lower() == "yes"
address = "AA" if is_measuring else "AB"

if is_measuring:
    DEFAULT_DISTANCE = 0
    distance_cm = prompt_with_default("The two stations' distance in cm", DEFAULT_DISTANCE)

    DEFAULT_MAX_RUNNING_TIME = 30
    max_running_time_s = prompt_with_default("Measure for at most this amount of seconds (per payload size)", DEFAULT_MAX_RUNNING_TIME)

print("Establishing serial connection to transceiver...")
serial = serial_setup(port, address)
print("Done. Now measuring..." if is_measuring else "Done. Station set to acknowledge mode.")

try:
    if is_measuring:
        for idx, payload_size in enumerate(PAYLOAD_SIZES, start=1):
            print("\n" + "=" * 60)
            print(f"[{idx}/{len(PAYLOAD_SIZES)}] Measuring payload size: {payload_size} B")
            print("=" * 60)

            measurer = Measurer(distance_cm, payload_size, max_running_time_s, serial)
            measurer.measure()
            measurer.print_and_save_stats()

            # brief pause between runs to let buffers settle
            sleep(0.5)

        print("\nAll payload sizes measured. Results written to the 'results/' folder.")
    else:
        print("Acknowledging everything I receive...")
        while True:
            serial.read_until()
            sleep(0.01)

except KeyboardInterrupt:
    print("\nQuitting execution...")
    exit()
