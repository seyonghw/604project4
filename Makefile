# Default target runs full analysis (placeholder for now)
.PHONY: all
all: 
	@echo "Running full analysis (placeholder)."

# Print today's predictions (single CSV line) and exit
.PHONY: predictions
predictions:
	@python3 scripts/predict.py

# Re-download raw data (placeholder)
.PHONY: rawdata
rawdata:
	@echo "Re-downloading raw data (placeholder)."

# Clean everything except code + raw data
.PHONY: clean
clean:
	@echo "Cleaning intermediate files (placeholder)."
