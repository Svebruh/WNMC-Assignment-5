from time import sleep, time_ns
from scipy.spatial import distance
from serial import Serial
from byte_coders import *
from random import choices
from string import ascii_lowercase
from numpy import mean, std, sqrt
from scipy import stats
import os
import math


class Measurer():
  """An object that performs the measurements and logs them to console and an appropriately named file."""

  def __init__(self, distance_cm: int, payload_size: int, max_running_time_s: int, serial: Serial):
    self.distance_cm = distance_cm
    self.payload_size = payload_size
    self.running_time_ns = max_running_time_s * 1_000_000_000
    self.s = serial
    self.elapsed_time_ns = 0
    self.sent_payloads = 0
    self.acknowledged_payloads = 0
    self.delays = []
    self._wall_start_ns = None

  def receive(self):
    input = dec(self.s.read_until()).strip()
    if (input == "m[R,A]"):
      self.acknowledged_payloads += 1
    return input 

  def measure(self):
    payload_str = "".join(choices(ascii_lowercase, k=self.payload_size))
    bytes_to_send = enc(f"m[{payload_str}\0,AB]\n")
    self._wall_start_ns = time_ns()

    while self.sent_payloads < 30 and (time_ns() - self._wall_start_ns) < self.running_time_ns:
      sleep(0.01)                         # small pacing to avoid overrunning serial
      t_0 = time_ns()
      self.s.write(bytes_to_send)

      acked = False
      deadline_ns = t_0 + 10_000_000_000   # 10s per-packet watchdog

      # Wait for either: ACK (RTT), or m[D] (done w/o ACK), or timeout
      while time_ns() < deadline_ns:
        line = dec(self.s.read_until()).strip()
        if not line:
          continue

        # ACKs come as: s[R,A,...]
        if line.startswith("s["):
          body = line[2:-1] if line.endswith("]") else line[2:]
          parts = [p.strip() for p in body.split(",")]
          if len(parts) >= 2 and parts[0] == "R" and parts[1].startswith("A"):
            acked = True
            break
      self.sent_payloads += 1
      if acked:
        t_1 = time_ns()
        t = t_1 - t_0                   # RTT
        self.elapsed_time_ns += t
        self.delays.append(t)
        self.acknowledged_payloads += 1
      else:
        self.delays.append(math.nan)

    # Drain remaining input briefly
    while True:
        line = dec(self.s.read_until()).strip()
        if not line:
            break

  def print_and_save_stats(self):

    os.makedirs("results", exist_ok=True)

    wall_elapsed_s = (time_ns() - self._wall_start_ns) / 1e9 if self._wall_start_ns else 0.0
    success_rate = (self.acknowledged_payloads / self.sent_payloads) if self.sent_payloads else 0.0
    throughput = (self.acknowledged_payloads * self.payload_size / wall_elapsed_s) if wall_elapsed_s > 0 else 0.0

    # Convert to seconds and DROP NaNs/Infs for stats
    rtts_s_all = [d / 1e9 for d in self.delays]
    rtts_s = [x for x in rtts_s_all if x == x and math.isfinite(x)]  # x==x filters NaN; also drop inf

    if len(rtts_s) == 0:
        mean_delay_s = math.nan
        standard_deviation_delay_s = math.nan
        cl, cr = (math.nan, math.nan)
    else:
        mean_delay_s = mean(rtts_s)
        if len(rtts_s) > 1:
            standard_deviation_delay_s = std(rtts_s, ddof=1)
            cl, cr = stats.t.interval(
                confidence=0.98,
                df=len(rtts_s) - 1,
                loc=mean_delay_s,
                scale=standard_deviation_delay_s / sqrt(len(rtts_s)),
            )
        else:
            # exactly one valid RTT → std = 0, CI undefined
            standard_deviation_delay_s = 0.0
            cl, cr = (math.nan, math.nan)

    print("-" * 20)
    print(f"Distance: {self.distance_cm}cm")
    print(f"Wall time: {wall_elapsed_s:.3f}s")
    print(f"Acknowledged {self.acknowledged_payloads} / {self.sent_payloads} payloads.")
    print(f"=> Success rate: {success_rate*100:.1f}%")
    print(f"Throughput: {throughput:.3f} B/s")
    print(f"Mean RTT: {mean_delay_s if not math.isnan(mean_delay_s) else 'NaN'}s")
    print(f"Std RTT: {standard_deviation_delay_s if not math.isnan(standard_deviation_delay_s) else 'NaN'}s")
    print(f"98% CI (RTT): ({cl if not math.isnan(cl) else 'NaN'}, {cr if not math.isnan(cr) else 'NaN'})s")

    # Save in your original format (line 1 summary, line 2 raw delays in nanoseconds)
    with open(f"results/{self.distance_cm}cm-{self.payload_size}B-results.csv", "w") as out_file:
        out_file.write(", ".join(str(x) for x in [
            success_rate, throughput, mean_delay_s, standard_deviation_delay_s, cl, cr
        ]) + "\n")
        out_file.write(", ".join(str(x) for x in self.delays))