=======================
Contributing
=======================

These practices are subject to change based on the decisions of the team.

- Use clear and explicit variable names. The variable names are typically more verbose than those in fastai.
- Python code should be formatted using black with the settings in pyproject.toml. The maximum line length is 120 characters.
- Contributions should be commited to a new branch and will be merged with main only after tests and documentation are complete.


Testing
==================

- All tests must be passing before merging with the ``main`` branch.
- Tests are automatically included in the CI/CD pipeline using Github actions.

Git Commits
===========

We use the `git3moji <https://robinpokorny.github.io/git3moji/>`_ standard for expressive git commit messages. 
Use one of the following five short emojis at the start of your of your git commit messages:

- ``:zap:`` ⚡️ – Features and primary concerns
- ``:bug:`` 🐛 – Bugs and fixes
- ``:tv:``  📺 – CI, tooling, and configuration
- ``:cop:`` 👮 – Tests and linting
- ``:abc:`` 🔤 – Documentation

As far as possible, please keep your git commits granular and focussed on one thing at a time. 
Please cite an the number of a Github issue if it relates to your commit.

Documentation
==================

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


Code of Conduct
==================

Our Pledge
---------------------

We as members, contributors, and leaders pledge to make participation in our
community a harassment-free experience for everyone, regardless of age, body
size, visible or invisible disability, ethnicity, sex characteristics, gender
identity and expression, level of experience, education, socio-economic status,
nationality, personal appearance, race, caste, color, religion, or sexual
identity and orientation.

We pledge to act and interact in ways that contribute to an open, welcoming,
diverse, inclusive, and healthy community.

Our Standards
---------------------

Examples of behavior that contributes to a positive environment for our
community include:

* Demonstrating empathy and kindness toward other people
* Being respectful of differing opinions, viewpoints, and experiences
* Giving and gracefully accepting constructive feedback
* Accepting responsibility and apologizing to those affected by our mistakes,
  and learning from the experience
* Focusing on what is best not just for us as individuals, but for the overall
  community

Examples of unacceptable behavior include:

* The use of sexualized language or imagery, and sexual attention or advances of
  any kind
* Trolling, insulting or derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or email address,
  without their explicit permission
* Other conduct which could reasonably be considered inappropriate in a
  professional setting

Enforcement Responsibilities
----------------------------

Community leaders are responsible for clarifying and enforcing our standards of
acceptable behavior and will take appropriate and fair corrective action in
response to any behavior that they deem inappropriate, threatening, offensive,
or harmful.

Community leaders have the right and responsibility to remove, edit, or reject
comments, commits, code, wiki edits, issues, and other contributions that are
not aligned to this Code of Conduct, and will communicate reasons for moderation
decisions when appropriate.

Scope
----------------------------

This Code of Conduct applies within all community spaces, and also applies when
an individual is officially representing the community in public spaces.
Examples of representing our community include using an official e-mail address,
posting via an official social media account, or acting as an appointed
representative at an online or offline event.

Enforcement
----------------------------

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported to the community leaders responsible for enforcement by email.
All complaints will be reviewed and investigated promptly and fairly.

All community leaders are obligated to respect the privacy and security of the
reporter of any incident.

Enforcement Guidelines
----------------------------

Community leaders will follow these Community Impact Guidelines in determining
the consequences for any action they deem in violation of this Code of Conduct:

1. Correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Community Impact**: Use of inappropriate language or other behavior deemed
unprofessional or unwelcome in the community.

**Consequence**: A private, written warning from community leaders, providing
clarity around the nature of the violation and an explanation of why the
behavior was inappropriate. A public apology may be requested.

2. Warning
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Community Impact**: A violation through a single incident or series of
actions.

**Consequence**: A warning with consequences for continued behavior. No
interaction with the people involved, including unsolicited interaction with
those enforcing the Code of Conduct, for a specified period of time. This
includes avoiding interactions in community spaces as well as external channels
like social media. Violating these terms may lead to a temporary or permanent
ban.

3. Temporary Ban
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Community Impact**: A serious violation of community standards, including
sustained inappropriate behavior.

**Consequence**: A temporary ban from any sort of interaction or public
communication with the community for a specified period of time. No public or
private interaction with the people involved, including unsolicited interaction
with those enforcing the Code of Conduct, is allowed during this period.
Violating these terms may lead to a permanent ban.

4. Permanent Ban
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Community Impact**: Demonstrating a pattern of violation of community
standards, including sustained inappropriate behavior, harassment of an
individual, or aggression toward or disparagement of classes of individuals.

**Consequence**: A permanent ban from any sort of public interaction within the
community.

Attribution
-----------

This Code of Conduct is adapted from the `Contributor Covenant <https://www.contributor-covenant.org>`_,
version `2.1 <https://www.contributor-covenant.org/version/2/1/code_of_conduct.html>`_.
