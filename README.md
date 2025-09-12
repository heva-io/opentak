# OpenTAK Clustering

[![image](https://img.shields.io/pypi/v/opentak.svg)](https://pypi.python.org/pypi/opentak)
[![image](https://img.shields.io/pypi/l/opentak.svg)](https://github.com/heva-io/opentak/blob/main/LICENSE)
[![image](https://img.shields.io/pypi/pyversions/opentak.svg)](https://pypi.python.org/pypi/opentak)


<p align="center">
  <img src="docs/assets/logo.png" alt="Logo TAK" width="200"/>
</p>


**OpenTAK** is a python package for **clustering** and **visualizing** treatment sequences in a cohort. It aims to identify, cluster, and represent the different treatment sequences used, while quantifying the number of patients involved in each of these sequences.

Under the hood, it runs on a Hierarchical Clustering Algorithm.  

📖 Documentation: https://heva-io.github.io/opentak/latest/  
📝 Blog (methodology + real use cases): 
https://hevaweb.com/en/articles/tak-r-celebrates-its-4th-anniversary/120


## Installation

To get started with the package, run one of the following command:

```bash
pip install opentak
```
or:

```bash
poetry add opentak
```
or: 
```bash
uv add opentak
```

## Quick Start

Starting from an event log with 3 columns — `ID_PATIENT`, `EVT`, and `TIMESTAMP` (int) — you can easily plot treatment sequences for each patient.  
The sequences are automatically ordered and clustered by similarity.  

👉 Check out the full [documentation](https://heva-io.github.io/opentak/latest/) for more details.

```python
from opentak import TakBuilder, TakVisualizer
from opentak.generation_cohort_tak import GenerateCohortTAK

NB_PATIENTS = 400
NB_JOURS_END = 370
n_clusters = 3

# Event log generation
evtlog = GenerateCohortTAK(
    nb_patients=NB_PATIENTS, nb_days_end=NB_JOURS_END, random_state=42
)
evtlog.initialisation_dataframe(
    treatment_name="Treatment A",
    dose_mean=int(NB_JOURS_END / 10),
    dose_std=int(NB_JOURS_END / 25),
)
evtlog.add_switch_gaussien("Treatment B")
evtlog = evtlog.add_in_out()
evtlog = evtlog.sort_values(by=["ID_PATIENT", "TIMESTAMP"])

# TAK builder 
tak = TakBuilder(evtlog).build()
tak.fit(n_clusters=n_clusters)

# TAK visualizer
tak_viz = TakVisualizer(tak)
tak_viz.process_visualization()
figplotly = tak_viz.get_plot(add_sep=True, unit_as_months=True, nb_months=2)
figplotly.update_layout(height=500, width=700)
figplotly.show()
```
You should obtain the following visualization.

<p align="center">
  <img src="docs/assets/tak_quickstart.png" alt="Logo TAK" width="600"/>
</p>

You can explore the `examples` folder to see additional applications on various event logs.

## Contributing
Contributions are welcome! To contribute:

1. Fork the repository and create a new branch for your changes.
2. Install the package dependencies using uv.  
First install uv by following the [official documentation guide](https://docs.astral.sh/uv/getting-started/installation/). Then run:
    ```
    uv sync --all-groups
    ```
3. Ensure your code is well-documented and includes relevant tests.

4. Checklist before submitting a pull request:

     - All tests pass with pytest
     - Ruff reports no linting errors when you run:
        ```
        uv run ruff format .
        uv run ruff check .
        ```
    - Mypy reports no type errors when you run:
        ```
        uv run mypy opentak
        ```

5. Open a pull request with a clear description of your changes and reference any related issues when possible.


## Acknowledgements
- Big thanks to [Marie Laurent](https://www.linkedin.com/in/marie-laurent-656727134/) for kicking off the idea and building the first version of the package. 
- Shout-out to [Alexandre Batisse](https://www.linkedin.com/in/alexandre-batisse-401578b4/), [Martin Prodel](https://www.linkedin.com/in/prodelmartin/), [Hugo de Oliveira](https://www.linkedin.com/in/hugo-de-oliveira/), and all former contributors from the [Heva](https://hevaweb.com/en) Data Science team for their feedback, refactoring, and feature enhancements. 
- And of course, cheers to all future contributors who will keep pushing this project forward.