import os
import numpy as np

import torch
import pickle

from femtorun import FemtoRunner
from femtodriverpub import SPURunner, FakeSPURunner

from femtocrux import CompilerClient, FQIRModel
import zipfile

import logging

import argparse
from argparse import RawTextHelpFormatter # allow carriage returns in help strings, for displaying model options

import yaml

MODELDIR = "../models"

def model_helpstr():
    yamlfname = f'{MODELDIR}/options.yaml'
    with open(yamlfname, 'r') as file:
        model_desc = yaml.safe_load(file)

    s = f"symlink the google drive folder to femtodriver/femtodriver/models ({MODELDIR})\n"
    s += "available models:\n"
    thisdir, subdirs, files = next(iter(os.walk(MODELDIR)))
    for file in files:
        if file.endswith('.pt'):
            modelname = file[:-3]

            s += f"  {modelname}"
            if modelname not in model_desc:
                s += f"\t  <-- missing specification in options.yaml"
            s += "\n"
                
    return s

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
            description="run a pickled FASMIR or FQIR on hardware. Compare with output of FB's SimRunner")

    parser.add_argument("model",
        help="path relative to models/ directory to .pt model to run\n" + model_helpstr())
    parser.add_argument("--n_inputs", default=2,
        help="number of random inputs to drive in")
    parser.add_argument("--runner", default='fakezynq',
        help="primary runner to use: (options: zynq, fakezynq)")
    parser.add_argument("--comparisons", default='hw,fasmir',
        help="runners to compare against, comma-separated. Options: hw, fasmir, fqir, fmir")
    parser.add_argument("--debug", default=False, action='store_true',
        help="set debug log level")
    parser.add_argument("--norun", default=False, action='store_true',
        help="just init (which will capture APB records for SD card), don't actually run aynthing")

    args = parser.parse_args()

    if not args.norun:
        raise NotImplementedError("does not support running over the wire + comparison yet. Coming in a future release")

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    ##################################
    # find the model in  options.yaml, load it

    # grab the models' pickle and its yaml description
    _, subdirs, files = next(iter(os.walk(MODELDIR)))

    model = args.model
    if model.startswith(MODELDIR):
        model = model[len(MODELDIR) + 1:]
    if model.endswith('.pt'):
        model = model[:-3]

    modelpt = model + '.pt'

    yamlfname = f'{MODELDIR}/options.yaml'
    ptfname = f'{MODELDIR}/{modelpt}'

    # open yaml
    with open(yamlfname, 'r') as file:
        model_desc = yaml.safe_load(file)

    if modelpt in files:
        # open model
        model_obj = torch.load(ptfname)
        if model_obj.__class__.__name__ not in ['FASMIR', 'GraphProto']:
            raise RuntimeError(f"supplied model {ptfname} didn't contain FASMIR or FQIR")

    else:
        raise RuntimeError(f"DIDN'T PROVIDE A VALID MODEL NAME\nyou had : {args.model} (={model})\n\n{model_helpstr()}")

    if model_obj.__class__.__name__ != 'GraphProto':
        raise NotImplementedError("femtodriverpub only works with FQIR")


    compiler_kwargs = {}
    if model in model_desc:
        if 'compiler_kwargs' in model_desc[model]:
            compiler_kwargs = model_desc[model]['compiler_kwargs']

    ################################
    # run the femtocrux docker to get memory images

    client = CompilerClient()

    bitstream = client.compile(    
        FQIRModel(
            model_obj,
            batch_dim = 0,
            sequence_dim = 1,
        )
    )

    # Write to a file for later use
    with open('images.zip', 'wb') as f: 
        f.write(bitstream)

    IMAGE_PATH = 'docker_data'
    # unzip it
    if not os.path.exists(IMAGE_PATH):
        os.mkdir(IMAGE_PATH)
    with zipfile.ZipFile("images.zip","r") as zip_ref:
        zip_ref.extractall("docker_data")
    
    ################################
    # run the SPU, or generate SD programming files

    if args.runner == "zynq":
        runner_cls = SPURunner
        runner_kwargs = {'platform' : 'zcu104', 'image_fname_pre' : 'test'}
    if args.runner == "redis":
        runner_cls = SPURunner
        runner_kwargs = {'platform' : 'redis', 'image_fname_pre' : 'test'}
    elif args.runner == "fakezynq":
        runner_cls = FakeSPURunner
        runner_kwargs = {'image_fname_pre' : 'test'}

    hw_runner = runner_cls(IMAGE_PATH, compiler_kwargs=compiler_kwargs, **runner_kwargs)

    ################################
    # if norun, just call reset, which generates SD files
    if args.norun:
        hw_runner.reset()
        exit()

    ################################
    # otherwise, run the SPU alongside any comparisons requested

    compare_runners = []
    compare_names = []
    for comp in comparisons:
        if comp == 'hw':
            compare_runners.append(hw_runner)
            compare_names.append('hardware')
        elif comp == 'fasmir':
            pass # TODO, connect to FX
        elif comp == 'fqir':
            pass # TODO, connect to FX
        elif comp == 'fmir':
            pass # TODO, connect to FX

    assert False # should not reach here in this release

    comparisons = args.comparisons.split(',')

    if len(comparisons) > 1:
        N = args.n_inputs
        inputs = hw_runner.make_fake_inputs(N)

        internals = FemtoRunner.compare_runs(
            inputs, 
            *compare_runners, 
            names=compare_names, 
            except_on_error=False)
    else:
        output_vals, internal_vals, _ = compare_runners[0].run(inputs)

