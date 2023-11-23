from water_tracker import WaterTracker

def main():
    wt = WaterTracker()
    wt.compute_standardized_indicator_levels_in_france("spi", save=True)

if __name__ == "__main__":
    main()
