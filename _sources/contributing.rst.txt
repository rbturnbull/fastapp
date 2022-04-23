=======================
Contributing
=======================


These practices are subject to change based on the decisions of the team.

- Use clear and explicit variable names. The variable names are typically more verbose than those in fastai.
- Python code should be formatted using black with the settings in pyproject.toml. The maximum line length is 120 characters.
- Add emojis at the start of git commit messages according to the `git3moji specification <https://robinpokorny.github.io/git3moji/>`_.
- Contributions should be commited to a new branch and will be merged with main only after tests and documentation are complete.

Testing
-------

- Tests must be added for any new rules before merging to the ``main`` branch. 
- All tests must be passing before merging with the ``main`` branch.
- Tests are automatically included in the CI/CD pipeline using Github actions.

Documentation
-------------

- Docstrings for Python functions should use the Google docstring convention (https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Citations for dependencies or research used in any rule should be added in bibtex format to ``docs/references.bib``.
- Documentation generated using sphinx and automatically deployed as part of the CI/CD pipeline.
- Docs should be written in reStructuredText.

Files need to start with a heading for the section. The convention used here is to use the equals sign above and below the heading::

    ===============
    Section Heading
    ===============

Subsections also use an equals sign but just below the heading::

    Subsection Heading
    ==================

Subsubsections have a single dash below the heading::

    Subsubsection Heading
    ---------------------

Try not to have any other sections within this but if it is necessary, use tildas below the heading::

    Further Subsection Headings
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

Other information for using reStructuredText in Sphinx can be found here: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#rst-primer and https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html.





