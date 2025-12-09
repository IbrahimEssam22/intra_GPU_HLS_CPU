# HEVC Intra Prediction HLS Makefile
# Comprehensive build system for HLS synthesis and validation

# Configuration
PROJECT = hls_intra_predict_prj
SOLUTION = solution1
TOP_FUNCTION = hls_intra_predict_top
TARGET_DEVICE = xczu9eg-ffvb1156-2-e
CLOCK_PERIOD = 7.3
VITIS_HLS = $(HOME)/vivado_install/Vitis_HLS/2021.1/bin/vitis_hls

# File dependencies
SOURCE_FILES = hls_intra_predict.h hls_intra_predict.cpp
TEST_FILES = hls_testbench.cpp allMats.c
SCRIPT_FILES = hls_synthesis.tcl run_hls_flow.sh
VALIDATION_FILES = validate_hls.c

# Default target
all: validate synthesize

# Help target
help:
	@echo "HEVC Intra Prediction HLS Build System"
	@echo "====================================="
	@echo ""
	@echo "Available targets:"
	@echo "  validate    - Run standalone validation against allMats.c"
	@echo "  synthesize  - Run complete HLS synthesis flow"
	@echo "  clean       - Clean all generated files"
	@echo "  setup       - Setup HLS project and run synthesis"
	@echo "  check       - Check HLS synthesis results"
	@echo "  reports     - Generate comprehensive reports"
	@echo "  all         - Run validation + synthesis"
	@echo ""
	@echo "Requirements:"
	@echo "  - Vitis HLS 2021.1 at ~/vivado_install/Vitis_HLS/2021.1/bin/"
	@echo "  - GCC compiler for validation"
	@echo ""

# Validation target
validate: validate_hls validation_output.txt
	@echo "✓ Validation completed - check validation_output.txt"

validate_hls: validate_hls.c allMats.c
	gcc -o validate_hls validate_hls.c -lm

validation_output.txt: validate_hls
	./validate_hls > validation_output.txt

# HLS synthesis target
synthesize: $(PROJECT)/$(SOLUTION)/syn/report/$(TOP_FUNCTION)_csynth.rpt
	@echo "✓ HLS synthesis completed"

$(PROJECT)/$(SOLUTION)/syn/report/$(TOP_FUNCTION)_csynth.rpt: $(SOURCE_FILES) $(TEST_FILES) hls_synthesis.tcl
	$(VITIS_HLS) -f hls_synthesis.tcl

# Complete setup target
setup: validate
	./run_hls_flow.sh

# Check synthesis results
check: $(PROJECT)/$(SOLUTION)/syn/report/$(TOP_FUNCTION)_csynth.rpt
	@echo "=== HLS Synthesis Results ==="
	@echo "Latency:"
	@grep -A3 "worst case" $(PROJECT)/$(SOLUTION)/syn/report/$(TOP_FUNCTION)_csynth.rpt || true
	@echo ""
	@echo "Timing:"
	@grep -A1 "Clock Period" $(PROJECT)/$(SOLUTION)/syn/report/$(TOP_FUNCTION)_csynth.rpt || true
	@echo ""
	@echo "Resources:"
	@grep -A10 "Utilization Estimates" $(PROJECT)/$(SOLUTION)/syn/report/$(TOP_FUNCTION)_csynth.rpt || true

# Generate reports
reports: HLS_PROJECT_SUMMARY.txt validation_output.txt
	@echo "✓ Reports generated:"
	@echo "  - HLS_PROJECT_SUMMARY.txt"
	@echo "  - validation_output.txt"
	@echo "  - $(PROJECT)/$(SOLUTION)/syn/report/ (HLS reports)"

# Clean targets
clean:
	rm -rf $(PROJECT)
	rm -f validate_hls
	rm -f validation_output.txt
	rm -f HLS_PROJECT_SUMMARY.txt
	rm -f *.log
	@echo "✓ Cleaned all generated files"

clean-project:
	rm -rf $(PROJECT)
	@echo "✓ Cleaned HLS project"

# Phony targets
.PHONY: all help validate synthesize setup check reports clean clean-project

# File targets
HLS_PROJECT_SUMMARY.txt: $(PROJECT)/$(SOLUTION)/syn/report/$(TOP_FUNCTION)_csynth.rpt
	./run_hls_flow.sh 2>/dev/null || true