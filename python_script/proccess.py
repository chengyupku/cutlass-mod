prefix = "../logs/logs_4x4/"
for gmem_slowdown in range(1, 11):
    for dsmem_acc in range(1, 7):
        output_file_name = prefix + f"log_gmemslowdown_{gmem_slowdown}x_dsmem_{dsmem_acc}x.txt"

        with open(output_file_name, "w") as output_file:
            for n in range(1,8):
                input_file_name = prefix + f"log_gmemslowdown_{gmem_slowdown}x_dsmem{dsmem_acc}x_gpu_{n}.txt"

                with open(input_file_name, "r") as input_file:
                    content = input_file.read()
                    output_file.write(content)
        print("Done")