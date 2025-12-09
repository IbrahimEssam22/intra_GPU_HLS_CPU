open_project hls_intra_predict_prj_stream
set_top hls_intra_predict_top
add_files hls_intra_predict.cpp
add_files hls_intra_predict.h
add_files -tb hls_testbench_updated.cpp
add_files -tb allMats.c

open_solution "solution1" -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default

# CRITICAL CHANGE: Remove ap_ce pin
config_interface -clock_enable=0

csynth_design
export_design -format ip_catalog
exit