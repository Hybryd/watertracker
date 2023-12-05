from water_tracker import WaterTracker

def main():
    indicateur = "nappes profondes"

    wt = WaterTracker()
    #wt.process(indicateur=indicateur)
    
    # # Then
    wt.load()
    wt.plot_counts_france(indicateur=indicateur)
    # wt.compute_standardized_indicator_values(indicateur=indicateur, freq="M", scale=3)
    # wt.aggregate_standardized_indicator_means_last_year(indicateur=indicateur)
    # wt.save()

if __name__ == "__main__":
    main()
