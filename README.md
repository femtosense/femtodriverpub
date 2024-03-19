# Femtosense Femtodriver (public : "femtodriverpub")

Using the memory images emitted by femtocrux, creates an SD card programming file that the firmware uses.

## Installation:

in the base directory of the repo:

```
pip install -e .
```

If you just want to run `sd_from_femtocrux.py`, and not `run_from_pt.py` (which requires femtocrux),
you may remove or comment out the femtocrux requirement in `femtodriverpub/PY_REQUIREMENTS.txt`. This will avoid having to install femtocrux's dependencies including torch.

See other Femtosense documentation for instructions how to install femtocrux.
You will be prompted to download the FX docker image the first time you run.
Remember to start docker daemon and add yourself to the "docker" group. E.g.:

```
sudo systemctl start docker
sudo usermod -aG docker <username>
```

## Usage:

Important note: if you are using TC2, you must set the `FS_HW_CFG` environment variable to `spu1p2v1.dat`.
If you are using the mass production chip, you must set it to `spu1p3v1.dat`.

E.g. add `export FS_HW_CFG=spu1p3v1.dat` to your bashrc. 

#### To generate SD programming files from a previously generated memory image zip

first unpack the memory image .zip emitted by femtocrux.

in `femtodriverpub/run/`:

```
python sd_from_femtocrux.py <path-to-unzipped-femtocrux-output-directory>
```

this will populate `io_records/apb_records/`, which has the `0PROG_A` and `0PROG_D` files which can be downloaded to the SD card. Note that future firmware might allow multiple models to coexist on the SD card. The leading '0' indicates that this is the first model.

#### To generate SD programming files from a previously saved FQIR pickle

This currently only works with the PyTorch flow. For now, for TF use `sd_from_femtocrux.py`.

You can pickle the docker's input, the FQIR graph, with `torch.save()` (In the pytorch/TF femtocrux walkthroughs, this variable is called `fqir_graph`). Put these pickles in `femtodriverpub/models/`.

`run_from_pt.py` works on these pickles, invoking the femtocrux docker to compile them and produce a zip containing memory images. It then does the same thing that `sd_from_femtocrux.py` does, producing SD card programming files. 

For now, you must supply the `--norun` option (in the future, this is the script that will be used to stream inputs/get ouptuts directly from a cable-connected SPU board).

Example:

```
run_from_pt.py ../models/identity.pt --norun
```

Notice the `fx_compiled.zip` that appears and was unpacked to `model_datas/identity/meta_from_femtocrux/`. 
`model_datas/identity/io_records/apb_records` contains the PROG files.

Pickles are notoriously unportable. Ideally, any pickling/unpickling is done on one machine, but failing that, try to ensure the pickle is unpacked using the same tool versions it was generated with.

## Future

Updates are planned that will:
- allow a direct connection `PC <--(USB)--> host <--(SPI)--> SPU`
    - this will allow for direct issuing of SPI commands from the PC to the SPU
    - it will also enable side-by-side comparison of the SPU and the golden model running inside the docker container
- allow creation of a more compact programming file for the EVK's SPI flash
- have a proper serialization format for the FQIR pickle

