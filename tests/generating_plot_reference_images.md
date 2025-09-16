# Generating references images for plot tests
** The reference images must be generated on a Mac computer**

1) Install the required dependency:
```bash
conda install pytest-mpl -c conda-forge
```

2) Navigate to the tests folder where the test_plots.py file is located


3) Generate the reference images by running:
```bash
pytest --mpl-generate-path=plot_reference_images/ -m mpl_image_compare
```

4) Verify that the tests pass by running:
```bash
pytest test_plots.py --mpl-baseline-path=eyedentify/tests/plot_reference_images
```


