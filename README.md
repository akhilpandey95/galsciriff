# galsciriff
Data, scripts, and recipes for fine-tuning GALACTICA large language model for SciRIFF dataset

### About
Using GALACTICA models to run experiments on all tasks across SciRIFF dataset.

### Environment setup
```shell
pip install -q wandb trl transformers peft
pip install -q -U bitsandbytes
```

### Run
**Linux:**
```shell
/usr/bin/time -v python galai_ft.py
```

**MacOS:**
```shell
/usr/bin/time -l -h python galai_ft.py
```

### License
[MIT License](https://github.com/akhilpandey95/LMRSD/blob/main/LICENSE)

### Authors and Collaborators
[Akhil Pandey](https://github.com/akhilpandey95)
