"""Tiny helper: build a notebook from a list of ('md'|'code', source) cells."""
import nbformat


def build(path, cells):
    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    for kind, src in cells:
        src = src.strip('\n')
        nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == 'md' else nbformat.v4.new_code_cell(src))
    nbformat.write(nb, path)
