import os
import sys
import numpy as np

import torch
import pickle

import femtodriverpub
from femtorun import FemtoRunner, DummyRunner
from femtodriverpub import SPURunner, FakeSPURunner
from femtodriverpub.fx_runner import FXRunner

from femtodriverpub.program_handler import ProgramHandler

import zipfile

try:
    from femtobehav.sim.runner import SimRunner  # for comparison
    from femtomapper.run import FQIRArithRunner, FMIRRunner

    DEV_MODE = True
except ImportError:
    DEV_MODE = False

import logging

import argparse
from argparse import (
    RawTextHelpFormatter,
)  # allow carriage returns in help strings, for displaying model options

import yaml

from femtodriverpub.run.util import process_single_outputs
from pathlib import Path


TOP_LEVEL_PACKAGE_DIR = Path(femtodriverpub.__file__).parent
MODELDIR = TOP_LEVEL_PACKAGE_DIR / Path("models")
MODELDIR = str(MODELDIR)


def check_dev_mode(feat):
    if not DEV_MODE:
        raise RuntimeError(
            f"{feat} is a FS-only feature, requires internal packages. Exiting"
        )


def model_helpstr(modeldir=MODELDIR):
    yamlfname = f"{modeldir}/options.yaml"
    with open(yamlfname, "r") as file:
        model_desc = yaml.safe_load(file)

    s = "\navailable models in femtodriverpub/femtodriverpub/models:\n"
    thisdir, subdirs, files = next(iter(os.walk(modeldir)))
    for file in files:
        if file.endswith(".pt"):
            modelname = file[:-3]

            s += f"  {modelname}"
            if modelname not in model_desc:
                s += f"\t  <-- missing specification in options.yaml"
            s += "\n"
        elif file.endswith(".pck"):
            modelname = file[:-4]

            s += f"  {modelname}"
            if modelname not in model_desc:
                s += f"\t  <-- missing specification in options.yaml"
            s += "\n"

    return s


def get_options_path(modeldir, model_options_file):
    if model_options_file is not None:
        model_options_file = os.path.expanduser(model_options_file)
        if not os.path.exists(model_options_file):
            raise ValueError(
                f"supplied model options file {model_options_file} does not exist"
            )
        return model_options_file

    else:
        return os.path.join(modeldir, "options.yaml")


def load_model_options(model, options_path):
    """look up the options (just compiler kwargs right now) for the model"""

    # open yaml to get model options
    with open(options_path, "r") as file:
        model_desc = yaml.safe_load(file)

    if "DEFAULT" in model_desc:
        print("found DEFAULT compiler options")
        compiler_kwargs = model_desc["DEFAULT"]["compiler_kwargs"]
    else:
        compiler_kwargs = {}

    if model in model_desc:
        if "compiler_kwargs" in model_desc[model]:
            compiler_kwargs.update(model_desc[model]["compiler_kwargs"])

    print("loaded the following compiler options")
    if "DEFAULT" in model_desc:
        print("(based on DEFAULT options)")

    for k, v in compiler_kwargs.items():
        print(f"  {k} : {v}")

    return compiler_kwargs


def load_model(model_path):
    """try to find the model, make what you get when you unpack matches the extension"""

    if not os.path.exists(model_path):
        raise ValueError("supplied model file {model_path} does not exist")

    model_with_ext = os.path.basename(os.path.expanduser(model_path))
    model, model_ext = os.path.splitext(model_with_ext)

    if model_ext in [".pt", ".pth"]:
        # open model
        fqir = torch.load(model_path, map_location=torch.device("cpu"))
        if fqir.__class__.__name__ not in ["FASMIR", "GraphProto"]:
            raise RuntimeError(
                f"supplied model {ptfname} didn't contain FASMIR or FQIR"
            )
        fasmir = None
    elif model_ext == ".pck":
        # open model
        fasmir = pickle.load(open(model_path, "rb"))
        if fasmir.__class__.__name__ not in ["FASMIR", "GraphProto"]:
            raise RuntimeError(
                f"supplied model {ptfname} didn't contain FASMIR or FQIR"
            )
        fqir = None

    else:
        raise ValueError(
            f"invalid model extension. Got {model_ext}. Need one of: .pt/.pth (FQIR pickle) or .pck (FASMIR pickle)"
        )

    return model, fasmir, fqir


def main(argv, modeldir=MODELDIR):
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="run a pickled FASMIR or FQIR on hardware. Compare with output of FB's SimRunner\n\n"
        + "Useful recipes:\n"
        + "----------------------\n"
        + "Run on hardware, default comparisons with full debug (fasmir):\n"
        + "\tpython run_from_pt.py ../models/modelname --debug --debug_vars=all\n\n"
        + "Generate SD (no board/cable needed):\n"
        + "\tpython run_from_pt.py ../models/modelname --runner=fakezynq --norun\n\n"
        + "Run simulator (no board/cable needed, ignore the comparison):\n"
        + "\tpython run_from_pt.py ../models/modelname --runner=fakezynq\n\n",
    )

    parser.add_argument(
        "model",
        help="model to run. " + model_helpstr(),
    )
    parser.add_argument(
        "--model_options_file",
        default=None,
        help=".yaml with run options for different models (e.g. compiler options). Default is femtodriverpub/femtodriverpub/models/options.yaml",
    )
    parser.add_argument(
        "--output_dir",
        default="model_datas",
        help="where to write fasmir, fqir, programming images, programming streams, etc",
    )
    parser.add_argument(
        "--n_inputs",
        default=2,
        type=int,
        help="number of random sim inputs to drive in",
    )
    parser.add_argument(
        "--input_file",
        default=None,
        help="file with inputs to drive in. Expects .npy from numpy.save. Expecting single 2D array of values, indices are (timestep, vector_dim)",
    )
    parser.add_argument(
        "--input_file_sample_indices",
        default=None,
        help="lo,hi indices to run from input_file",
    )
    parser.add_argument(
        "--force_femtocrux_compile",
        default=False,
        action="store_true",
        help="force femtocrux as the compiler, even if FS internal packages present",
    )
    parser.add_argument(
        "--force_femtocrux_sim",
        default=False,
        action="store_true",
        help="force femtocrux as the simulator, even if FS internal packages present",
    )
    parser.add_argument(
        "--runner",
        default="zynq",
        help="primary runner to use: (options: zynq, fakezynq, redis)",
    )
    parser.add_argument(
        "--comparisons",
        default="hw,fasmir",
        help="runners to compare against, comma-separated. Options: hw, fasmir, fqir, fmir, fakehw",
    )
    parser.add_argument(
        "--debug_vars",
        default=None,
        help="debug variables to collect and compare values for, comma-separated (no spaces), or 'all'",
    )
    parser.add_argument(
        "--debug_vars_fname",
        default=None,
        help="file with a debug variable name on each line",
    )
    parser.add_argument(
        "--debug", default=False, action="store_true", help="set debug log level"
    )
    parser.add_argument(
        "--norun",
        default=False,
        action="store_true",
        help="just init (which will capture APB records for SD card), don't actually run aynthing",
    )
    parser.add_argument(
        "--noencrypt",
        default=False,
        action="store_true",
        help="don't encrypt programming files",
    )
    parser.add_argument(
        "--sim_est_input_period",
        default=None,
        type=float,
        help="simulator input period for energy estimation. No impact on runtime. Floating point seconds",
    )
    parser.add_argument(
        "--dummy_output_file",
        default=None,
        help="for fakezynq, the values that the runner should reply with. Specify a .npy for a single variable",
    )

    args = parser.parse_args(argv)

    if args.debug:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        # turn these down, they're long and annoying
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
        PIL_logger = logging.getLogger("PIL")
        PIL_logger.setLevel(logging.WARNING)
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # grab the models' pickle and its yaml description
    model_options_path = get_options_path(modeldir, args.model_options_file)
    model, fasmir, fqir = load_model(args.model)
    compiler_kwargs = load_model_options(model, model_options_path)

    # collect debug vars
    debug_vars = []
    if args.debug_vars_fname is not None:
        varf = open(args.debug_vars_fname, "r")
        debug_vars += varf.readlines()

    if args.debug_vars is not None:
        debug_vars += args.debug_vars.split(",")

    comparisons = args.comparisons.split(",")

    if "fqir" in comparisons or "fmir" in comparisons:
        if fqir is None:
            raise RuntimeError(
                "asked for fqir or fmir comparison, but did't start from FQIR"
            )

    # primary runner

    runner_kwargs = {"encrypt": not args.noencrypt}
    if args.runner == "zynq":  # hard SPU plugged into FPGA
        runner_kwargs["platform"] = "zcu104"
        runner_kwargs["program_pll"] = True
        runner_kwargs["fake_connection"] = False

    elif args.runner == "fpgazynq":  # soft SPU inside FPGA logic
        runner_kwargs["platform"] = "zcu104"
        runner_kwargs["program_pll"] = False
        runner_kwargs["fake_connection"] = False

    elif args.runner == "redis":  # redis-based simulation (questa)
        runner_kwargs["platform"] = "redis"
        runner_kwargs["program_pll"] = True
        runner_kwargs["fake_connection"] = False

    elif args.runner == "fakezynq":  # e.g. for generating EVK program
        runner_kwargs["platform"] = "zcu104"
        runner_kwargs["program_pll"] = False
        runner_kwargs["fake_connection"] = True

    elif args.runner == "fakeredis":  # e.g. for integration test
        runner_kwargs["platform"] = "redis"
        runner_kwargs["program_pll"] = False
        runner_kwargs["fake_connection"] = True

    else:
        raise RuntimeError(f"Unknown runner {args.runner}")

    ################################
    # run!

    # make SPURunner and SimRunner to compare it to
    fake_hw_recv_vals = None
    if args.dummy_output_file is not None:
        fake_hw_recv_vals = np.load(args.dummy_output_file)

    if DEV_MODE and not args.force_femtocrux_compile:
        compiler = "femtomapper"
    else:
        compiler = "femtocrux"

    model_dir = os.path.join(os.path.expanduser(args.output_dir), f"{model}")
    meta_dir = os.path.join(model_dir, f"meta_from_{compiler}")
    io_records_dir = os.path.join(model_dir, "io_records")

    for dirpath in (meta_dir, io_records_dir):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    handler = ProgramHandler(
        fasmir=fasmir,
        fqir=fqir,
        compiler=compiler,
        compiler_kwargs=compiler_kwargs,
        encrypt=not args.noencrypt,
        data_dir=meta_dir,
    )

    hw_runner = SPURunner(
        meta_dir,
        fake_hw_recv_vals=fake_hw_recv_vals,
        debug_vars=debug_vars,
        **runner_kwargs,
        io_records_dir=io_records_dir,
    )

    if DEV_MODE:
        hw_runner.attach_debugger(handler.fasmir)

        with open(os.path.join(model_dir, "fasmir.txt"), "w") as f:
            f.write(str(handler.fasmir))
        if fqir is not None:
            with open(os.path.join(model_dir, "fqir.txt"), "w") as f:
                f.write(str(fqir))

    # fill in for 'all' debug vars option
    # not all runners can necesarily take 'all' as a debug vars arg
    if args.debug_vars == "all" or args.debug_vars == ["all"]:
        debug_vars = hw_runner.debug_vars

    # for fake runner, what do you reply with
    if args.dummy_output_file is not None:
        fname = args.dummy_output_file
        if fname.endswith(".npy"):
            dummy_vals = np.load(fname)
            dummy_output_dict = {hw_runner.get_single_output_name(): dummy_vals}
        elif fname.endswith(".pt"):
            # would put dictionary with multiple output vars in here
            raise RuntimeError(
                "unsupported file format for --dummy_output_file, only .npy is supported"
            )
        else:
            raise RuntimeError(
                "unsupported file format for --dummy_output_file, only .npy is supported"
            )
    else:
        dummy_output_dict = None

    compare_runners = []
    compare_names = []

    if args.force_femtocrux_compile and not args.force_femtocrux_sim:
        check_dev_mode("FX compile but dev mode sim")
        # have to make our own FASMIR so we can simulate it
        # this is a little iffy, weakly relies on compiler determinism
        # since we have compiled twice here
        # even a nondeterministic compiler should always be correct, though
        unused_meta_dir = f"{model}_unused_fx_compile_but_not_sim"
        parallel_dev_handler = ProgramHandler(
            fasmir=None,
            fqir=fqir,
            compiler="femtomapper",
            compiler_kwargs=compiler_kwargs,
            encrypt=not args.noencrypt,
            data_dir=unused_meta_dir,
        )
        sim_fasmir = parallel_dev_handler.fasmir
        sim_fmir = parallel_dev_handler.fmir
    else:
        sim_fasmir = handler.fasmir
        sim_fmir = handler.fmir

    for comp in comparisons:
        if comp == "hw":
            compare_runners.append(hw_runner)
            compare_names.append("hardware")
        elif comp == "fasmir":
            if DEV_MODE and not args.force_femtocrux_sim:
                # FB runner
                fasmir_runner = SimRunner(
                    sim_fasmir,  # model might be fqir, need to compile for SimRunner
                    input_padding=hw_runner.io.input_padding,
                    output_padding=hw_runner.io.output_padding,
                )

            else:
                # use FXRunner which wraps docker
                fasmir_runner = FXRunner(
                    handler.fqir,  # XXX it will recompile, not sure if there's a way to get it to use what it already compiled
                    input_padding=hw_runner.io.input_padding,
                    output_padding=hw_runner.io.output_padding,
                )

            compare_runners.append(fasmir_runner)
            compare_names.append("fasmir")

        elif comp == "fqir":
            check_dev_mode("comparison to FQIR runner")
            # FIXME, could move FQIRRunner def from FM to fmot
            if fqir is not None:
                fq_runner = FQIRArithRunner(fqir)
                compare_runners.append(fq_runner)
                compare_names.append("fqir")
        elif comp == "fmir":
            check_dev_mode("comparison to FMIR runner")
            if args.force_femtocrux_compile and args.force_femtocrux_sim:
                raise NotImplementedError("FX can't simulate FMIR")
            if fqir is not None:
                fm_runner = FMIRRunner(sim_fmir)
                compare_runners.append(fm_runner)
                compare_names.append("fmir")
        elif comp == "dummy":
            fakehw_runner = DummyRunner(dummy_output_dict)
            compare_runners.append(fakehw_runner)
            compare_names.append("dummy")
        else:
            raise RuntimeError(f"unknown comparison runner '{comp}'")

    # make some fake inputs, or load from file

    N = args.n_inputs

    if args.input_file is None:
        inputs = hw_runner.make_fake_inputs(N, style="random")
    else:
        if args.input_file.endswith(".npy"):
            input_vals = np.load(args.input_file)
            N = input_vals.shape[0]
            inputs = hw_runner.make_fake_inputs(N, style="random")
            if len(inputs) > 1:
                raise RuntimeError("can only support one input via file")
            for k, v in inputs.items():
                inputs[k] = input_vals

            # trim to sample range, if supplied
            if args.input_file_sample_indices is not None:
                lo, hi = args.input_file_sample_indices.split(",")
                for k, v in inputs.items():
                    inputs[k] = inputs[k][int(lo) : int(hi)]

        else:
            raise RuntimeError(
                "unsupported file format for --input_file, only .npy is supported"
            )

    if len(inputs) == 1:
        print("single input variable detected, saving to inputs.npy")
        np.save("inputs.npy", inputs[next(iter(inputs))])

    # if norun, just reset and exit
    if args.norun:
        hw_runner.reset()
        exit()

    hw_runner.ioplug.start_recording("io_sequence")

    if len(comparisons) > 1:
        compare_status = {}
        outputs, internals = FemtoRunner.compare_runs(
            inputs,
            *compare_runners,
            names=compare_names,
            compare_internals=len(debug_vars) > 0,
            except_on_error=False,
            compare_status=compare_status,
        )

    else:
        compare_runners[0].reset()
        output_vals, internal_vals, _ = compare_runners[0].run(inputs)
        outputs = {compare_names[0]: output_vals}
        internals = {compare_names[0]: internal_vals}

    hw_runner.ioplug.commit_recording("all.yaml")

    pickle.dump(inputs, open(os.path.join(meta_dir, "runner_inputs.pck"), "wb"))
    pickle.dump(outputs, open(os.path.join(meta_dir, "runner_outputs.pck"), "wb"))
    pickle.dump(internals, open(os.path.join(meta_dir, "runner_internals.pck"), "wb"))

    print(
        f"outputs and internal variables pickles saved to {os.path.join(meta_dir, 'runner_*.pck')}"
    )
    print("  unpickle with internals = pickle.load(open('runner_internals.pck', 'rb'))")
    print("  then internals[runner_name][varname][j]")
    print("  is runner_name's values for varname at timestep j")
    print("  fasmir, fmir, fqir will report everything.")
    print("  the setting of --debug_vars determines what's available from hardware.")

    out_fnames, _ = process_single_outputs(outputs)
    if out_fnames is not None:
        print(
            f"also saved single output variable's values for each runner to {out_fnames}"
        )
        print("  summarized to output_diff.png")

    for runner in compare_runners:
        if runner.__class__.__name__ == 'SimRunner':
            print(
                f"found a SimRunner, sending metrics to {os.path.join(model_dir, 'metrics.txt')}"
            )
            metrics_str = runner.get_metrics(
                as_str=True, input_period=args.sim_est_input_period
            )
            metrics = runner.get_metrics(input_period=args.sim_est_input_period)
            with open(os.path.join(model_dir, "metrics.txt"), "w") as f:
                f.writelines(metrics_str)
            print("power was", metrics["Power (W)"])

    # repeat output comparison result
    print()
    if len(comparisons) > 1:
        if compare_status["pass"]:
            print("===================================")
            print("comparison good!")
            print("===================================")
        else:
            print(compare_status["status_str"])
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("COMPARISON FAILED")
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            exit(1)
    else:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("comparison not performed")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


if __name__ == "__main__":
    main(sys.argv[1:])
