import pandas as pd
from io import StringIO

# Provided log data
log_data = """
gmem_slowdown: 1x, dsmem_accelerate: 0x, Max_GFLOPS: 756121.0, Kernel: 0_gemm_codegen
gmem_slowdown: 1x, dsmem_accelerate: 1x, Max_GFLOPS: 756121.0, Kernel: 0_gemm_codegen
gmem_slowdown: 1x, dsmem_accelerate: 2x, Max_GFLOPS: 756177.0, Kernel: 3_gemm_codegen
gmem_slowdown: 1x, dsmem_accelerate: 3x, Max_GFLOPS: 764051.0, Kernel: 9_gemm_codegen
gmem_slowdown: 1x, dsmem_accelerate: 4x, Max_GFLOPS: 777600.0, Kernel: 23_gemm_codegen
gmem_slowdown: 1x, dsmem_accelerate: 5x, Max_GFLOPS: 770369.0, Kernel: 2_gemm_codegen
gmem_slowdown: 1x, dsmem_accelerate: 6x, Max_GFLOPS: 775089.0, Kernel: 27_gemm_codegen
gmem_slowdown: 2x, dsmem_accelerate: 0x, Max_GFLOPS: 354532.0, Kernel: 0_gemm_codegen
gmem_slowdown: 2x, dsmem_accelerate: 1x, Max_GFLOPS: 621946.0, Kernel: 5_gemm_codegen
gmem_slowdown: 2x, dsmem_accelerate: 2x, Max_GFLOPS: 668514.0, Kernel: 6_gemm_codegen
gmem_slowdown: 2x, dsmem_accelerate: 3x, Max_GFLOPS: 687376.0, Kernel: 6_gemm_codegen
gmem_slowdown: 2x, dsmem_accelerate: 4x, Max_GFLOPS: 702901.0, Kernel: 26_gemm_codegen
gmem_slowdown: 2x, dsmem_accelerate: 5x, Max_GFLOPS: 712178.0, Kernel: 26_gemm_codegen
gmem_slowdown: 2x, dsmem_accelerate: 6x, Max_GFLOPS: 724681.0, Kernel: 44_gemm_codegen
gmem_slowdown: 3x, dsmem_accelerate: 0x, Max_GFLOPS: 209710.0, Kernel: 0_gemm_codegen
gmem_slowdown: 3x, dsmem_accelerate: 1x, Max_GFLOPS: 415039.0, Kernel: 21_gemm_codegen
gmem_slowdown: 3x, dsmem_accelerate: 2x, Max_GFLOPS: 537200.0, Kernel: 32_gemm_codegen
gmem_slowdown: 3x, dsmem_accelerate: 3x, Max_GFLOPS: 623480.0, Kernel: 32_gemm_codegen
gmem_slowdown: 3x, dsmem_accelerate: 4x, Max_GFLOPS: 637065.0, Kernel: 32_gemm_codegen
gmem_slowdown: 3x, dsmem_accelerate: 5x, Max_GFLOPS: 659214.0, Kernel: 30_gemm_codegen
gmem_slowdown: 3x, dsmem_accelerate: 6x, Max_GFLOPS: 666493.0, Kernel: 30_gemm_codegen
gmem_slowdown: 4x, dsmem_accelerate: 0x, Max_GFLOPS: 157583.0, Kernel: 0_gemm_codegen
gmem_slowdown: 4x, dsmem_accelerate: 1x, Max_GFLOPS: 366671.0, Kernel: 40_gemm_codegen
gmem_slowdown: 4x, dsmem_accelerate: 2x, Max_GFLOPS: 478760.0, Kernel: 30_gemm_codegen
gmem_slowdown: 4x, dsmem_accelerate: 3x, Max_GFLOPS: 536220.0, Kernel: 49_gemm_codegen
gmem_slowdown: 4x, dsmem_accelerate: 4x, Max_GFLOPS: 564751.0, Kernel: 49_gemm_codegen
gmem_slowdown: 4x, dsmem_accelerate: 5x, Max_GFLOPS: 582783.0, Kernel: 36_gemm_codegen
gmem_slowdown: 4x, dsmem_accelerate: 6x, Max_GFLOPS: 589216.0, Kernel: 36_gemm_codegen
gmem_slowdown: 5x, dsmem_accelerate: 0x, Max_GFLOPS: 126546.0, Kernel: 0_gemm_codegen
gmem_slowdown: 5x, dsmem_accelerate: 1x, Max_GFLOPS: 328161.0, Kernel: 45_gemm_codegen
gmem_slowdown: 5x, dsmem_accelerate: 2x, Max_GFLOPS: 428234.0, Kernel: 49_gemm_codegen
gmem_slowdown: 5x, dsmem_accelerate: 3x, Max_GFLOPS: 497052.0, Kernel: 49_gemm_codegen
gmem_slowdown: 5x, dsmem_accelerate: 4x, Max_GFLOPS: 514268.0, Kernel: 49_gemm_codegen
gmem_slowdown: 5x, dsmem_accelerate: 5x, Max_GFLOPS: 536548.0, Kernel: 49_gemm_codegen
gmem_slowdown: 5x, dsmem_accelerate: 6x, Max_GFLOPS: 544660.0, Kernel: 49_gemm_codegen
gmem_slowdown: 6x, dsmem_accelerate: 0x, Max_GFLOPS: 105517.0, Kernel: 0_gemm_codegen
gmem_slowdown: 6x, dsmem_accelerate: 1x, Max_GFLOPS: 291378.0, Kernel: 32_gemm_codegen
gmem_slowdown: 6x, dsmem_accelerate: 2x, Max_GFLOPS: 391554.0, Kernel: 29_gemm_codegen
gmem_slowdown: 6x, dsmem_accelerate: 3x, Max_GFLOPS: 447228.0, Kernel: 49_gemm_codegen
gmem_slowdown: 6x, dsmem_accelerate: 4x, Max_GFLOPS: 472761.0, Kernel: 36_gemm_codegen
gmem_slowdown: 6x, dsmem_accelerate: 5x, Max_GFLOPS: 494223.0, Kernel: 49_gemm_codegen
gmem_slowdown: 6x, dsmem_accelerate: 6x, Max_GFLOPS: 495043.0, Kernel: 49_gemm_codegen
gmem_slowdown: 7x, dsmem_accelerate: 0x, Max_GFLOPS: 89583.0, Kernel: 0_gemm_codegen
gmem_slowdown: 7x, dsmem_accelerate: 1x, Max_GFLOPS: 269336.0, Kernel: 49_gemm_codegen
gmem_slowdown: 7x, dsmem_accelerate: 2x, Max_GFLOPS: 360410.0, Kernel: 49_gemm_codegen
gmem_slowdown: 7x, dsmem_accelerate: 3x, Max_GFLOPS: 402454.0, Kernel: 49_gemm_codegen
gmem_slowdown: 7x, dsmem_accelerate: 4x, Max_GFLOPS: 424493.0, Kernel: 49_gemm_codegen
gmem_slowdown: 7x, dsmem_accelerate: 5x, Max_GFLOPS: 431671.0, Kernel: 49_gemm_codegen
gmem_slowdown: 7x, dsmem_accelerate: 6x, Max_GFLOPS: 441452.0, Kernel: 49_gemm_codegen
gmem_slowdown: 8x, dsmem_accelerate: 0x, Max_GFLOPS: 79050.0, Kernel: 0_gemm_codegen
gmem_slowdown: 8x, dsmem_accelerate: 1x, Max_GFLOPS: 254195.0, Kernel: 36_gemm_codegen
gmem_slowdown: 8x, dsmem_accelerate: 2x, Max_GFLOPS: 341247.0, Kernel: 49_gemm_codegen
gmem_slowdown: 8x, dsmem_accelerate: 3x, Max_GFLOPS: 379054.0, Kernel: 49_gemm_codegen
gmem_slowdown: 8x, dsmem_accelerate: 4x, Max_GFLOPS: 398790.0, Kernel: 49_gemm_codegen
gmem_slowdown: 8x, dsmem_accelerate: 5x, Max_GFLOPS: 409434.0, Kernel: 49_gemm_codegen
gmem_slowdown: 8x, dsmem_accelerate: 6x, Max_GFLOPS: 417051.0, Kernel: 49_gemm_codegen
gmem_slowdown: 9x, dsmem_accelerate: 0x, Max_GFLOPS: 70249.0, Kernel: 0_gemm_codegen
gmem_slowdown: 9x, dsmem_accelerate: 1x, Max_GFLOPS: 237542.0, Kernel: 49_gemm_codegen
gmem_slowdown: 9x, dsmem_accelerate: 2x, Max_GFLOPS: 317862.0, Kernel: 49_gemm_codegen
gmem_slowdown: 9x, dsmem_accelerate: 3x, Max_GFLOPS: 352400.0, Kernel: 49_gemm_codegen
gmem_slowdown: 9x, dsmem_accelerate: 4x, Max_GFLOPS: 365251.0, Kernel: 49_gemm_codegen
gmem_slowdown: 9x, dsmem_accelerate: 5x, Max_GFLOPS: 374208.0, Kernel: 49_gemm_codegen
gmem_slowdown: 9x, dsmem_accelerate: 6x, Max_GFLOPS: 377847.0, Kernel: 49_gemm_codegen
gmem_slowdown: 10x, dsmem_accelerate: 0x, Max_GFLOPS: 63060.0, Kernel: 0_gemm_codegen
gmem_slowdown: 10x, dsmem_accelerate: 1x, Max_GFLOPS: 227209.0, Kernel: 49_gemm_codegen
gmem_slowdown: 10x, dsmem_accelerate: 2x, Max_GFLOPS: 301515.0, Kernel: 49_gemm_codegen
gmem_slowdown: 10x, dsmem_accelerate: 3x, Max_GFLOPS: 332506.0, Kernel: 49_gemm_codegen
gmem_slowdown: 10x, dsmem_accelerate: 4x, Max_GFLOPS: 342268.0, Kernel: 49_gemm_codegen
gmem_slowdown: 10x, dsmem_accelerate: 5x, Max_GFLOPS: 351250.0, Kernel: 49_gemm_codegen
gmem_slowdown: 10x, dsmem_accelerate: 6x, Max_GFLOPS: 352490.0, Kernel: 49_gemm_codegen
"""

# Creating a DataFrame from the log data
log_df = pd.read_csv(StringIO(log_data), sep=", ", engine='python', header=None)
log_df.columns = ['gmem_slowdown', 'dsmem_accelerate', 'Max_GFLOPS', 'Kernel']

# Extracting content after the colon in each column
log_df = log_df.applymap(lambda x: x.split(': ')[1])
log_df['gmem_slowdown'] = log_df['gmem_slowdown'].apply(lambda x: f"1/{x.split('x')[0]}")
log_df['dsmem_accelerate'] = log_df['dsmem_accelerate'].apply(lambda x: '0x (no noc)' if x == '0x' else x)
log_df['Kernel'] = log_df['Kernel'].apply(lambda x: x.split('_')[0])

excel_path = 'log_data.xlsx'
log_df.to_excel(excel_path, index=False)