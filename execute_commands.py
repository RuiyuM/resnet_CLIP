import subprocess
import time
import itertools

def execute_commands_from_file(file_path, commands_at_a_time=3, wait_time=30):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Group the commands into chunks of size commands_at_a_time
    command_groups = [lines[i:i+commands_at_a_time] for i in range(0, len(lines), commands_at_a_time)]

    for group in command_groups:
        processes = []
        for command in group:
            print(f"Executing command: {command.strip()}")
            process = subprocess.Popen(command, shell=True)
            processes.append(process)

        # Wait for all processes in the group to finish
        for process in processes:
            process.wait()

        print(f"Finished executing group of commands. Waiting for {wait_time} minutes...")
        time.sleep(60 * wait_time)

if __name__ == "__main__":
    execute_commands_from_file("baseline.txt")
