import subprocess
import time


def execute_commands_from_file(file_path, num_gpus):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Execute first `num_gpus` commands concurrently
    processes = []
    for i in range(num_gpus):
        if i < len(lines):
            command = lines[i].strip()
            print(f"Executing command: {command}")
            process = subprocess.Popen(command, shell=True, text=True)
            processes.append(process)

    # Process the remaining commands
    for command in lines[num_gpus:]:
        # Wait for any process to complete
        finished_process_index, _ = subprocess.wait(processes, return_when=subprocess.FIRST_COMPLETED)
        finished_process = processes.pop(finished_process_index)
        print(f"Finished executing command.")

        # Execute the next command on the available GPU
        command = command.strip()
        print(f"Executing command: {command}")
        process = subprocess.Popen(command, shell=True, text=True)
        processes.append(process)

    # Wait for the remaining processes to complete
    for process in processes:
        process.wait()


if __name__ == "__main__":
    num_gpus = 8
    execute_commands_from_file("baseline.txt", num_gpus)
