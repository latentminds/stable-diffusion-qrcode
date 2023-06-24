import sys
import subprocess


def main():
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/holwech/diffusers.git",
        ]
    )


if __name__ == "__main__":
    main()
