# Thermal moonquake supporting code

Companion code for the paper "Thermal moonquake characterization and cataloging using frequency-based algorithms and stochastic gradient descent" submitted to JGR Planets. 

Items include:
- *convert_data_to_physunits.py*: Converts the data from uncompressed volts to physical units
- *find_synthetic_azimuth.py*: Generates a synthetic moonquake and finds the azimuth direction for the event. Can be used to determine the optimal stochastic gradient descent parameters as described in the paper. 
- *fine_tune_detection.py*: Code to finetune the detection obtained from the Civilini et al. 2021 catalog. 
- *extract_evids.py*: Extracts a trace around the event time of the original Civilini et al. 2021 catalog and saves it as a pickle file. Required for the fine-tuning code. 
- *resp.pkl*: Instrument response curves obtained from Nunn et al. [2020] (DOI: https://doi.org/10.1007/s11214-020-00709-3)

Note:
Civilini et al. [2021] catalog (DOI: https://doi.org/10.1093/gji/ggab083) from the supplementary materials of that paper is **required** for *extract_evids.py* and *fine_tune_detection.py*. Thbe article is open-access and the supplementary files can be found at the bottom of the webpage. 
