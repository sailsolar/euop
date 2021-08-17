import argparse
import blackSholes as bS

if __name__ == "__main__":
    eu_call = bS.EuropeanCall(100, 100, 0.3, 1, 0.01)
    print(f"European Call Option price as per BS Model: {eu_call.price}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", "-p", required=False, dest="packages",
                        type=argparse.FileType(mode="r", encoding="utf-8"),
                        help="Path to packages paths file. Use '-' to read packages paths from stdin")
    args = vars(parser.parse_args())
