import os
import sys

import sitecustomize  # noqa: F401
from streamlit.web import bootstrap


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(project_root, "app.py")
    sys.path.insert(0, project_root)
    bootstrap.run(script_path, False, [], {})


if __name__ == "__main__":
    main()
