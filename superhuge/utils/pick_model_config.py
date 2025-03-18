import os
import time
from rich import print
from rich.console import Console
from rich.table import Table
import threading


def pick_file(root_path: str | None):
    """Lists YAML files in the default directory and allows the user to choose one using CLI."""
    console = Console()
    default_dir = os.path.join(
        root_path if root_path else os.getcwd(), "configs", "models"
    )

    if not os.path.exists(default_dir):
        print(
            f"\n[bold red]Error:[/bold red] The directory {default_dir} does not exist."
        )
        return None

    yaml_files = [f for f in os.listdir(default_dir) if f.endswith(".yaml")]

    if not yaml_files:
        print("\n[bold yellow]No YAML files found.[/bold yellow]")
        return None

    table = Table(title="\nAvailable YAML Files")
    table.add_column("Index", justify="center", style="cyan", no_wrap=True)
    table.add_column("Filename", style="magenta")

    for i, file in enumerate(yaml_files, start=1):
        table.add_row(str(i), file)

    console.print(table)

    while True:
        try:
            choice = input("Enter the number of the file you want to select: ")
            if not choice.strip():
                return os.path.join(default_dir, yaml_files[0])
            choice = int(choice) - 1
            if 0 <= choice < len(yaml_files):
                return os.path.join(default_dir, yaml_files[choice])
            else:
                print("[bold red]Invalid selection, please try again.[/bold red]")
        except ValueError:
            print("[bold red]Please enter a valid number.[/bold red]")


# Example usage
if __name__ == "__main__":
    selected_file = pick_file(None)
    if selected_file:
        print(f"[bold green]You selected:[/bold green] {selected_file}")
    else:
        print("[bold yellow]No file selected.[/bold yellow]")
