| bucket | ms/step | % |
|---|---|---|
| eltwise | 460.5 | 38.6% |
| gemm | 218.9 | 18.4% |
| flash_attn | 210.9 | 17.7% |
| other | 132.8 | 11.1% |
| reduce | 78.7 | 6.6% |
| nccl | 55.0 | 4.6% |
| layernorm | 27.5 | 2.3% |
| memcpy | 7.4 | 0.6% |
| **TOTAL** | **1191.7** | **100%** |

comm_hidden_ratio: 0.961
compute_stream_busy_ms_per_step: 1136.7
comm_stream_busy_ms_per_step: 55.0