import subprocess
import time


def execute_commands_from_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    for command in lines:
        print(f"Executing command: {command.strip()}")
        subprocess.run(command, shell=True, check=True, text=True)
        print(f"Finished executing command. Waiting for 30 minutes...")
        time.sleep(60 * 80)


if __name__ == "__main__":
    execute_commands_from_file("baseline.txt")
