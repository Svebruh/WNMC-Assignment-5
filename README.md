# WNMC Assignment 5 — Visible Light Communication (VLC)

Minimal Python tools for ETH WNMC Assignment 05:

- **Chat CLI** (Step 4)
- **Measurement runner** (Step 5)
- **Plot generator** (report figures)

## Requirements

- Python 3.9+
- Install: `pip install pyserial numpy scipy pandas matplotlib`
- Baud rate: **115200**
- Serial ports: Windows `COM4`, `COM5`, … / macOS `/dev/tty.usb*` / Linux `/dev/ttyACM0`

## Step 4 — Chat (broadcast & unicast)

Run on two terminals/machines (example on Windows):

```
python chat_cli.py --port COM4 --address AB --nickname Alice
python chat_cli.py --port COM5 --address CD --nickname Bob
```

- **Broadcast** destination: `FF`
- **Unicast** to the peer’s address.
- VLC payload format: `m[<text>\0,<DEST>]` — note the **textual `\0`** terminator.

## Step 5 — Measurements

From `Measurments/`:

```
python main.py
```

- The measuring station records **RTT = send → first ACK (s[R,A,...])**.
- It runs **1, 100, 180 B** sequentially.
- CSVs saved to `results/` as `<distance>cm-<payload>B-results.csv` (summary line + raw RTTs in seconds).

### Plots

From `Measurments/`:

```
python plot_results.py
```

Outputs to `plots/`:

- Throughput vs distance (per payload & combined)
- Mean RTT vs distance **with CI**
- RTT **standard deviation** vs distance
- Boxplot summary of RTT distributions
