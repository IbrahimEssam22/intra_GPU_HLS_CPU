# HEVC IP Integration Checklist

## Quick Start Checklist

### ✅ Vivado Block Design Steps

1. **Create Project**
   - Target: xc7z020clg484-1 (Zynq-7000)
   - Add IP repository: `hls_intra_predict_prj/solution1/impl/`

2. **Add Required IP Blocks**
   - [ ] ZYNQ7 Processing System
   - [ ] AXI Direct Memory Access  
   - [ ] hls_intra_predict_top (your custom IP)
   - [ ] AXI Interconnect (auto-added)

3. **Configure Zynq PS**
   - [ ] Enable M_AXI_GP0 (AXI master port)
   - [ ] Enable FCLK_CLK0 at 100 MHz
   - [ ] Enable UART1 for debug output
   - [ ] Configure DDR3 memory

4. **Configure AXI DMA**
   - [ ] Disable Scatter-Gather (simple mode)
   - [ ] Set MM2S Data Width: **1024 bits** (power-of-2 requirement)
   - [ ] Set S2MM Data Width: **1024 bits** (power-of-2 requirement)
   - [ ] Both channels same width for simplicity
   - [ ] Buffer Length Register: 23 bits

5. **Make Connections**
   ```
   Control Path (AXI-Lite):
   Zynq PS M_AXI_GP0 → AXI Interconnect → [DMA S_AXI_LITE, HEVC IP s_axi_control]
   
   Data Path (AXI-Stream):  
   DMA M_AXIS_MM2S → HEVC IP input_stream_V
   HEVC IP output_stream_V → DMA S_AXIS_S2MM
   
   Clock/Reset:
   Zynq PS FCLK_CLK0 → [All IP clock inputs]
   Zynq PS FCLK_RESET0_N → [All IP reset inputs]
   ```

6. **Address Assignment** 
   - [ ] Auto-assign addresses (Window → Address Editor)
   - [ ] Verify no overlaps or conflicts

7. **Generate Hardware**
   - [ ] Validate Design (Tools → Validate Design)
   - [ ] Generate Block Design
   - [ ] Create HDL Wrapper
   - [ ] Generate Bitstream (15-30 minutes)
   - [ ] Export Hardware with bitstream → `hevc_system.xsa`

### ✅ Vitis IDE Project Setup

1. **Create Platform**
   - [ ] File → New → Platform Project
   - [ ] Import XSA: `hevc_system.xsa`  
   - [ ] Generate platform

2. **Create Application**
   - [ ] File → New → Application Project
   - [ ] Select your platform
   - [ ] Processor: ps7_cortexa9_0
   - [ ] Template: Empty Application (C)

3. **Add Source Files**
   - [ ] Copy `vitis_main_app.c` to src/ folder
   - [ ] Copy `test_data_setup.c` to src/ folder
   - [ ] Ensure BSP includes DMA and custom IP drivers

4. **Build Configuration**
   - [ ] Build project (should complete without errors)
   - [ ] Check for missing drivers in BSP

### ✅ Testing Strategy

**Phase 1: QEMU Emulation**
- [ ] Right-click project → Run As → Launch on Emulator
- [ ] Check basic software flow (may show stub results)
- [ ] Debug any compilation/linking issues

**Phase 2: Hardware Testing**
- [ ] Connect JTAG cable to Zynq board
- [ ] Power on board
- [ ] Right-click project → Run As → Launch Hardware  
- [ ] Monitor UART output (115200 baud)
- [ ] Should see validation results

### ✅ Expected Output

```
HEVC Intra Prediction IP Test Application
Platform initialized

Initializing AXI DMA...
DMA initialization completed successfully
Initializing HEVC Intra Prediction IP...
HEVC IP initialization completed successfully

=== Starting HEVC Block Processing ===
Starting HEVC IP...
Starting DMA transfers...
Waiting for DMA transfers to complete...
Waiting for HEVC IP to complete...
HEVC block processing completed successfully!

Validating results...
=== HEVC Intra Prediction Results ===
Mode | Expected | Actual | Status
-----|----------|--------|-------
   0 |      50  |    50  | PASS
   1 |      43  |    43  | PASS
   ...
  34 |      92  |    92  | PASS

=== Summary ===
Total modes: 35
Passed: 35
Failed: 0
Success rate: 100.0%
*** ALL TESTS PASSED! IP working correctly. ***

=== Best Mode Analysis ===
Best mode: 1
Best cost: 43
Mode type: DC

*** SUCCESS: All tests passed! ***
HEVC IP is working correctly.
```

### ✅ Troubleshooting

**Common Issues & Solutions:**

1. **"DMA config lookup failed"**
   - Check XPARAMETERS_H defines
   - Verify DMA is included in block design
   - Regenerate BSP

2. **"DMA transfer timeout"** 
   - Check clock connections
   - Verify data width settings match
   - Check cache coherency settings

3. **"HEVC IP config lookup failed"**
   - Ensure custom IP driver is generated
   - Check IP is properly connected
   - Verify address assignment

4. **All costs return 0**
   - Check AXI-Stream connections
   - Verify IP is receiving data
   - Check reset is properly connected

5. **Build errors**
   - Refresh BSP and regenerate
   - Check all required drivers are included
   - Verify source file paths

### ✅ Performance Validation

**Expected Results:**
- Processing time: < 10ms per 8x8 block
- All 35 modes should pass validation
- Best mode typically: Mode 1 (DC) with cost 43
- No DMA timeouts or IP hangs

**Performance Monitoring:**
```c
// Add timing measurements
#include "xtime_l.h"

XTime start_time, end_time;
XTime_GetTime(&start_time);
process_hevc_block();
XTime_GetTime(&end_time);

double elapsed_us = (double)(end_time - start_time) / (COUNTS_PER_SECOND / 1000000);
xil_printf("Processing time: %.2f microseconds\n", elapsed_us);
```

### ✅ Next Steps

1. **Multi-block Processing**: Process multiple blocks in sequence
2. **Performance Optimization**: Pipeline transfers and processing
3. **Video Integration**: Connect to camera/display pipelines
4. **Error Handling**: Add robust error detection and recovery
5. **Interrupt Mode**: Use interrupts instead of polling for better efficiency

## Files Created

- `VIVADO_INTEGRATION_GUIDE.md` - Complete integration guide
- `test_data_setup.c` - Test data and validation functions  
- `vitis_main_app.c` - Main software application
- `HEVC_IP_CHECKLIST.md` - This checklist (current file)

Copy these files to your Vitis IDE project src/ folder and follow the checklist for successful integration!