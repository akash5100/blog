#!/usr/bin/env python
import os
import sys
import datetime

def handle_spaces(x: str) -> str:
    return x.replace(" ", "_")


def script():
    category = ""
    title = datetime.date.today().strftime("%Y-%m-%d") + "-"
    if len(sys.argv) <= 1:
        print('usage [-t "title"] [-c "category"] ')
        sys.exit(1)

    for i in range(len(sys.argv)):
        if sys.argv[i] == "-c":
            category = sys.argv[i+1]
        elif sys.argv[i] == "-t":
            x = sys.argv[i+1]

    # Create the new blog post file
    file_name = title + handle_spaces(x)

    file_path = os.path.join("_posts", file_name + ".markdown")

    with open(file_path, "w") as file:
        file.write("---\n")
        file.write(f"title: {x}\n")
        file.write(f"categories: {category}\n")
        file.write("---\n\n")
    file.close()
    print(f"Created\nnew blog post at \n'{file_path}' \nwith category '{category}'")


if __name__ == '__main__':
    script()