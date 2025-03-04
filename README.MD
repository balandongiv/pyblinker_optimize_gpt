---
title: MNE-Pyblinker: A Python package for automated eye-blink artifact detection in biosignals
tags:
  - Python
  - Biosignals
  - Neuroscience
  - Artifact Detection
  - Eye-blink
  - MNE-Python
  - EEG
  - EOG
  - Eye Aspect Ratio
authors:
  - name: [Your Name]
    orcid: [Your ORCID]
    affiliation: [Your Affiliation] # e.g., University of X
  - name: [Co-author Name (if any)]
    orcid: [Co-author ORCID (if any)]
    affiliation: [Co-author Affiliation (if any)]
date: 28 October 2024 # Date of submission
bibliography: paper.bib # if you have a bib file, otherwise remove this line
---

# Summary
pyblinker is a Python package primarily intended for automated detection of eye-blink artifacts in diverse biosignals, including EEG, EOG, and Eye Aspect Ratio (EAR). It addresses the limitations of existing MATLAB-based tools like BLINKERS, offering a Python-based, open-source solution for researchers analyzing various types of physiological data.

A key benefit of pyblinker is its enhanced functionality and broader applicability compared to BLINKERS.  By expanding input compatibility beyond EEG/EOG to include EAR, and by integrating seamlessly with the MNE-Python ecosystem, pyblinker provides a more versatile and accessible tool. This facilitates efficient and reproducible biosignal analysis workflows across neuroscience and related fields, ultimately streamlining research and improving data quality.
# Statement of need

Accurate analysis of eye-blink is paramount in biosignal research, providing crucial insights into physiological and cognitive processes across diverse modalities beyond traditional EEG, such as EOG and even video-based Eye Aspect Ratio (EAR).  However, manual identification and annotation of these artifacts is notoriously time-consuming, demands specialized expertise, and becomes a significant bottleneck, especially with the increasing scale of modern biosignal datasets.

The MATLAB-based BLINKERS (Kleifges et al., 2017 algorithm offers automated ocular index extraction specifically from Electroencephalography (EEG) and Electrooculography (EOG) data.  This MATLAB package automatically identifies blink-related components in EEG/EOG, calculates indices, generates reports, and provides summaries. Despite its sophistication for EEG/EOG, BLINKERS has limitations hindering broader use.

Firstly, BLINKERS' input is strictly limited to EEG or EOG channels, excluding valuable ocular measures like EAR derived from video eye-tracking, which are increasingly integrated into multimodal biosignal analysis.  Secodly, and critically for many researchers, BLINKERS' reliance on MATLAB creates a significant barrier to entry.  The requirement for a proprietary MATLAB license restricts access, particularly within the growing open-source and Python-centric scientific community, limiting the adoption of this otherwise valuable tool.


The scientific community, particularly in neuroscience and related fields, increasingly uses Python for biosignal analysis, with MNE-Python (Gramfort et al., 2013; Appelhoff et al., 2019) becoming a leading open-source platform. MNE-Python's open nature, extensive features, and community support make it ideal for EEG, MEG, and broader biosignal research. Integrating specialized tools into MNE-Python offers seamless workflows, access to diverse algorithms, coding best practices, and established biosignal analysis conventions. For example, MNE-Python's epoching is standard for time-windowed biosignal analysis related to events of interest. Integrating eye-blink detection within this framework allows straightforward derivation of artifact metrics within relevant epochs, crucial for many biosignal studies.

Furthermore, the trend towards agentic code development favors Python tools for automated workflows. A Python-based solution like pyblinker is more readily integrated into automated biosignal analysis pipelines than MATLAB-based alternatives.



# Code Quality and Documentation

pyblinker is hosted on GitHub[<sup>2</sup>](#fn2). Examples and API documentation are available on the platform [GitHub repository link will be inserted here] and Read the Docs[<sup>3</sup>](#fn3) [ReadTheDocs link will be inserted here]. We provide installation guides, algorithm introductions, and examples of using the package within the documentation. The package is designed to be cross-platform, available on Linux, macOS, and Windows for Python versions 3.9 - 3.11.  MNE-Pyblinker can be easily installed using pip via `pip install mne-pyblinker` [package name on PyPI will be inserted here].

To ensure high code quality and maintainability, all implementations adhere to the PEP8 code style (van Rossum et al., 2001), which is automatically enforced using `flake8`, the code formatter `black`, and the static analyzer `prospector` within our development workflow.  Comprehensive documentation is provided through docstrings following the NumPy conventions and is built using `Sphinx` to generate user-friendly HTML documentation. The codebase is accompanied by a suite of unit tests, currently achieving [XX]% code coverage, which are automatically executed in a continuous integration workflow upon each commit to ensure ongoing code reliability and prevent regressions.

# Acknowledgements

RPB wishes to thank the Ministry of Higher Education Malaysia for their financial support via the FRGS-EC 026-2024
# References

- Appelhoff, S., Sanderson, M., Brooks, T., Vliet, L., Quentin, R., Holdgraf, C., & Schirrmeister, R. T. (2019). MNE-Python: Software for Brain Data Analysis in Python. *Journal of Open Source Software*, *4*(43), 1074. [https://doi.org/10.21105/joss.01074](https://doi.org/10.21105/joss.01074)
- Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., Parkkonen, L., & Hämäläinen, M. S. (2013). MNE Python: Software for real-time and offline MEG/EEG analysis. *Frontiers in Neuroscience*, *7*, 267. [https://doi.org/10.3389/fnins.2013.00267](https://doi.org/10.3389/fnins.2013.00267)
- Kleifges, K., Fachner, J., Munte, T. F., & Bohr, K. (2017). Automatic artifact detection and ocular index computation using blind source separation. *Biological Psychology*, *128*, 132-141. [https://doi.org/10.1016/j.biopsycho.2017.07.007](https://doi.org/10.1016/j.biopsycho.2017.07.007)
- van Rossum, G., Warsaw, B., & Coghlan, A. (2001). PEP 8 -- Style Guide for Python Code. *Python Enhancement Proposals*. [https://peps.python.org/pep-0008/](https://peps.python.org/pep-0008/)

<div id="footnotes">
<sup>2</sup> GitHub repository URL will be inserted here.
<br>
<sup>3</sup> Read the Docs documentation URL will be inserted here.
</div>