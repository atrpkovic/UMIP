# main.py
from report import build_quarterly_report

if __name__ == "__main__":
    out = build_quarterly_report()
    print(f"Report written to: {out}")
