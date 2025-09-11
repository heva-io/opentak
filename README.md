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

To quickly get started with the package, run one of the following command:

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
import pandas as pd
from tak import TakBuilder, TakVisualizer

evtlog = pd.DataFrame(
    [
        [0, 0, "in"],
        [0, 0, "treatmentA"],
        [0, 2, "treatmentB"],
        [0, 6, "out"],
        [1, 0, "in"],
        [1, 0, "treatmentA"],
        [1, 3, "chemotherapy"],
        [1, 7, "death"],
        [2, 0, "in"],
        [2, 0, "treatmentA"],
        [2, 4, "chemotherapy"],
        [2, 8, "out"],
        [3, 0, "in"],
        [3, 0, "treatmentA"],
        [3, 5, "chemotherapy"],
        [3, 9, "out"],
        [4, 0, "in"],
        [4, 0, "treatmentA"],
        [4, 3, "treatmentB"],
        [4, 7, "out"],
        [6, 0, "in"],
        [6, 0, "treatmentA"],
        [6, 6, "chemotherapy"],
        [6, 10, "out"],
    ],
    columns=["ID_PATIENT","TIMESTAMP","EVT"],
)

tak = TakBuilder(evtlog).build()
tak.fit(n_clusters = 2)

tak_viz = TakVisualizer(tak)
tak_viz.process_visualization()

figplotly = tak_viz.get_plot(add_sep=True)
figplotly.update_layout(height=400, width=700)
figplotly.show()
```


## Contributing
Contributions are welcome! To contribute:

1. Fork the repository and create a new branch for your changes.
2. Install the package dependencies using uv:
    ```bash
    pip install uv
    ```
    ```
    uv sync --all-groups
    ```
3. Ensure your code is well-documented and includes relevant tests.

4. Checklist before submitting a pull request:

     - All tests pass with pytest
     - Ruff reports no linting errors when you run:
        ```
        ruff format .
        ruff check .
        ```
    - Mypy reports no type errors when you run:
        ```
        mypy opentak
        ```

5. Open a pull request with a clear description of your changes and reference any related issues when possible.


## Acknowledgements
- Big thanks to [Marie Laurent](https://www.linkedin.com/in/marie-laurent-656727134/) for kicking off the idea and building the first version of the package. 
- Shout-out to [Alexandre Batisse](https://www.linkedin.com/in/alexandre-batisse-401578b4/), [Martin Prodel](https://www.linkedin.com/in/prodelmartin/), [Hugo de Oliveira](https://www.linkedin.com/in/hugo-de-oliveira/), and all former contributors from the [Heva](https://hevaweb.com/en) Data Science team for their feedback, refactoring, and feature enhancements. 
- And of course, cheers to all future contributors who will keep pushing this project forward.