from datetime import date

def main():
    # Specs: 29 zones, 24 hours per zone = 696 load numbers
    # Then 29 peak-hour predictions (0–23), then 29 peak-day flags (0/1)
    n_zones = 29
    n_hours = 24
    loads = [0] * (n_zones * n_hours)      # 696 zeros
    peak_hours = [0] * n_zones             # 29 zeros
    peak_days = [0] * n_zones              # 29 zeros

    today_str = date.today().isoformat()   # "YYYY-MM-DD"

    # One CSV line: "YYYY-MM-DD", L1_00,...,L29_23, PH_1,...,PH_29, PD_1,...,PD_29
    # Values only (no headers), all integers.
    values = loads + peak_hours + peak_days
    # Print exactly one line, nothing else
    print('"{}", {}'.format(today_str, ", ".join(str(v) for v in values)))

if __name__ == "__main__":
    main()
