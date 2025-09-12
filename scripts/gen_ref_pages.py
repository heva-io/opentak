"""
Generate the code reference pages and navigation.
This scripts is based on mkdocstrings documentation:
https://mkdocstrings.github.io/recipes/#bind-pages-to-sections-themselves
"""

from pathlib import Path


import mkdocs_gen_files  # type: ignore[import-not-found]

nav = mkdocs_gen_files.Nav()

PACKAGE_NAME = "opentak"

root = Path(__file__).parent.parent
source_code = root / PACKAGE_NAME

for path in sorted(source_code.rglob("*.py")):
    module_path = path.relative_to(root).with_suffix("")
    doc_path = path.relative_to(source_code).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__" or parts[-1] == "__main__":
        continue

    nav[parts] = "../" + full_doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))


with mkdocs_gen_files.open("docs/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
