add_executable(nllb_moe_example nllb_moe_example.cc)
target_link_libraries(nllb_moe_example PUBLIC -lcublas -lcublasLt -lcudart
                      nllb_moe nvtx_utils mpi_utils nccl_utils cublasMMWrapper)