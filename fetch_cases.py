"""Download and verify the MATPOWER 8.1 inputs used by the study."""

import hashlib
from pathlib import Path
from urllib.request import urlopen


CASE_SHA256 = {
    "case9.m": "ee50fc7bf9f6019c0f3a3bc94d20978cc667b08f695dc725d00dbd998b358623",
    "case33bw.m": "b40831eeb444669ae876e2996f0dda9f05cd83e81b314b8dfca51e4890cca95d",
    "case69.m": "7bbdc8c39394eb6c924a64cad49d528bc8e99061823eb0f4a42b22da5c146a2c",
}
SOURCE = "https://raw.githubusercontent.com/MATPOWER/matpower/8.1/data"
DESTINATION = Path(__file__).resolve().parent / "work/simulation/data"


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, expected_hash in CASE_SHA256.items():
        target = DESTINATION / name
        if target.exists():
            data = target.read_bytes()
        else:
            with urlopen(f"{SOURCE}/{name}", timeout=30) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError(f"{target} does not match the archived MATPOWER 8.1 input")
        target.write_bytes(data)
        print(f"Verified {name}")


if __name__ == "__main__":
    main()
