import subprocess
import sys
import json

schedule_path = "schedules/schedule_4x4x1.json"
simulate_multiple_min = 0
simulate_multiple_max = 0
dsmem_accelerate_min = 1
dsmem_accelerate_max = 1

def run_command(command):
    with open('logs/run_log.txt', 'w') as f:
        try:
            result = subprocess.run(command, shell=True, check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"fail: {e}", file=f)
            f.flush()
            return False

def main():
    with open('logs/run_log.txt', 'w') as f:
        with open(schedule_path, 'r', encoding='utf-8') as sch_file:
            schedule_list = json.load(sch_file)
            print(f"test on {len(schedule_list)} schedules")
        for simulate_multiple in range(simulate_multiple_min, simulate_multiple_max + 1):
            for dsmem_accelerate in range(dsmem_accelerate_min, dsmem_accelerate_max + 1):
                print(f"simulating... simulate_multiple = {simulate_multiple}, dsmem_accelerate = {dsmem_accelerate}", file=f)
                f.flush()
                
                if not run_command("rm -rf ../build/examples/97_gemm_codegen/*_gemm_codegen"):
                    continue

                if not run_command(f"python codegen.py --simulate_multiple={simulate_multiple} --dsmem_accelerate {dsmem_accelerate} --schedule_path {schedule_path}"):
                    continue

                if not run_command("make -C ../build/examples/97_gemm_codegen/ -j 16"):
                    continue

                if not run_command(f"bash run.sh {simulate_multiple} {dsmem_accelerate} {len(schedule_list)}"):
                    continue
                    

                f.flush()

        print("Finish All", file=f)
        f.flush()

if __name__ == "__main__":
    main()