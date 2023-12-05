from water_tracker import WaterTracker

def main():
    indicateur = "nappes profondes"

    wt = WaterTracker()
    # First:
    #wt.process(indicateur=indicateur)
    
    # Then
    wt.load()
    wt.plot_counts_france(indicateur=indicateur)
    

if __name__ == "__main__":
    main()
