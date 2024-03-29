#!/usr/bin/env python3

import os
import sys
import datetime

def handle_spaces(x: str) -> str:
    return x.replace(" ", "_")


def script():
    tags = ""
    title = datetime.date.today().strftime("%Y-%m-%d") + "-"
    if len(sys.argv) <= 1:
        print('usage [-t "title"] [-g "tags"] ')
        sys.exit(1)

    for i in range(len(sys.argv)):
        if sys.argv[i] == "-g":
            tags = sys.argv[i+1]
        elif sys.argv[i] == "-t":
            x = sys.argv[i+1]

    # Create the new blog post file
    file_name = title + handle_spaces(x)
    file_path = os.path.join("_posts", file_name + ".md")

    # Create the assets folder
    assets_folder_name = handle_spaces(x)
    assets_folder_path = os.path.join("assets", assets_folder_name)
    os.makedirs(assets_folder_path, exist_ok=True)

    with open(file_path, "w") as file:
        file.write("---\n")
        file.write(f"title: {x}\n")
        file.write(f"tags: {tags}\n")
        file.write("---\n\n")
    file.close()

    print(f"Created\n> @'{file_path}' \n> tags: '{tags}'")
    print(f"Created assets folder\n> @'{assets_folder_path}'")

if __name__ == '__main__':
    script()
