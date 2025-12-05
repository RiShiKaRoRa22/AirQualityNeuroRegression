import pandas as pd

def calc_subindex(C, breakpoints):
    for (BP_lo, BP_hi, I_lo, I_hi) in breakpoints:
        if BP_lo <= C <= BP_hi:
            return ((I_hi - I_lo) / (BP_hi - BP_lo)) * (C - BP_lo) + I_lo
    return None

def calculate_AQI(row):
    bp_PM25 = [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,500,401,500)]
    bp_PM10 = [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(431,600,401,500)]
    bp_NO2 = [(0,40,0,50),(41,80,51,100),(81,180,101,200),(181,280,201,300),(281,400,301,400),(401,540,401,500)]
    bp_SO2 = [(0,40,0,50),(41,80,51,100),(81,380,101,200),(381,800,201,300),(801,1600,301,400),(1601,2600,401,500)]
    bp_CO = [(0,1,0,50),(1.1,2,51,100),(2.1,10,101,200),(10.1,17,201,300),(17.1,34,301,400),(34.1,51,401,500)]
    bp_O3 = [(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),(209,748,301,400),(749,1000,401,500)]
    bp_NH3 = [(0,200,0,50),(201,400,51,100),(401,800,101,200),(801,1200,201,300),(1201,1800,301,400),(1801,3000,401,500)]

    subindexes = []

    for pollutant, breaks in [
        ("PM2.5", bp_PM25), ("PM10", bp_PM10), ("NO2", bp_NO2),
        ("SO2", bp_SO2), ("CO", bp_CO), ("O3", bp_O3), ("NH3", bp_NH3)
    ]:
        if pd.notna(row[pollutant]):
            subindexes.append(calc_subindex(row[pollutant], breaks))

    # remove None entries
    subindexes = [x for x in subindexes if x is not None]

    #return max only if list has values
    return max(subindexes) if len(subindexes) > 0 else None
