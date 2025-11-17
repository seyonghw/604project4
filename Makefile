# Makefile

.PHONY: all rawdata predictions clean

# Default: run the SARIMAX fitting script
all:
	conda run -n pjm_env python -u fitting.py

# Re-download + re-merge raw data
rawdata:
	conda run -n pjm_env python -u load_data.py

# Print today's predictions
predictions:
	conda run -n pjm_env python prediction.py

# Clean fitted models and merged output
clean:
	rm -rf models
	rm -f output/merged_all_years.csv
