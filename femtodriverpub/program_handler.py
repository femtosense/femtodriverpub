import shutil
from femtodriverpub.typing_help import VARVALS, ARRAYU64, ARRAYINT, IOTARGET, HWTARGET
import femtodriverpub.util.packing as packing
import zipfile
import yaml
import os

import logging

logger = logging.getLogger(__name__)

# this is the fname preamble that comes out of FX,
# also use for when we compile from FM
# it's a little odd, yeah
MEM_IMAGE_FNAME_PRE = "test"


class ProgramHandler:
    """This makes all the program sources and compiler outputs look the same to SPURunner

    Knows how to accept these raw program inputs:
        - FQIR
        - FASMIR
        - zipfile from previous FX run

    Produces the two necessary SPURunner ingredients
        - an extracted data dir with the memory images
        - a metadata yaml

    Can compile/extract the following ways:
        - FQIR --(FX)--> zipfile --> images + meta
        - FQIR --(FM)--> FASMIR  --> images + meta + FASMIR(for debug)
    """

    def __init__(
        self,
        fqir=None,
        fasmir=None,
        zipfile_fname=None,
        compiler="femtocrux",
        encrypt=True,
        insert_debug_stores=False,
        data_dir="spu_runner_data",
        compiler_kwargs={},
    ):
        self.fqir = fqir
        self.fasmir = fasmir
        self.fmir = None  # can be filled in when compiling w/ FM
        self.zipfile_fname = zipfile_fname
        self.encrypt = encrypt
        self.insert_debug_stores = insert_debug_stores

        # this is where the outputs go
        # the runner uses the memory images and metadata here
        self.data_dir = data_dir

        # don't have a finished product, need to compile
        if self.fasmir is None and self.zipfile_fname is None:
            if self.fqir is None:
                raise ValueError("compilation must start with FQIR")

            if compiler == "femtocrux":
                self.zipfile_fname = "fx_compiled.zip"  # we will create this zipfile
                self._compile_with_fx_to_zipfile(compiler_kwargs=compiler_kwargs)
                self._extract_zipfile()

            elif compiler == "femtomapper":
                self._compile_with_fm_to_fasmir(mapper_conf_kwargs=compiler_kwargs)
                self._extract_fasmir()

        # start with already-generated FX output
        elif self.zipfile_fname is not None:
            self._extract_zipfile()

        # start with FASMIR object
        elif self.fasmir is not None:
            self._extract_fasmir()

        # modify the meta yaml with FQIR info
        # e.g. original shapes (for padding)
        if self.fqir is not None:
            self._add_fqir_meta()

    def _get_compiler_env_for_docker(self, docker_kwargs):
        # get compiler version
        hwcfg = os.getenv("FS_HW_CFG")
        if hwcfg is not None:
            if hwcfg == "spu1p3v1.dat":
                version = {"FS_HW_CFG": hwcfg}
            elif hwcfg == "spu1p2v1.dat":
                version = {"FS_HW_CFG": hwcfg}
            else:
                raise ValueError(
                    f"unknown FS_HW_CFG value {hwcfg}. Must be spu1p2v1.dat or spu1p3v1.dat for TC2 or MP chip, respectively"
                )
        else:
            # assume default of 1p3
            version = {"FS_HW_CFG": "spu1p3v1.dat"}
            logger.warning(
                "FS_HW_CFG not explicitly set. Assuming default of ISA 1p3v1 (mass production chip)"
            )
            logger.warning(
                "  set 'export FS_HW_CFG=spu1p3v1.dat' for mass production chip"
            )
            logger.warning("  set 'export FS_HW_CFG=spu1p2v1.dat' for TC2")
            yn = input(
                "enter 'y' if the default of ISA 1p3 is OK, otherwise set the environment variable and try again"
            )
            if yn != "y":
                exit(-1)

        docker_kwargs["environment"].update(version)

    def _compile_with_fx_to_zipfile(self, compiler_kwargs={}):
        """calls FX to turn fqir into the zipfile"""
        from femtocrux import CompilerClient, FQIRModel

        docker_kwargs = {"environment": {}}
        self._get_compiler_env_for_docker(docker_kwargs)  # get compiler version
        client = CompilerClient(docker_kwargs=docker_kwargs)

        bitstream = client.compile(
            FQIRModel(
                self.fqir,
                batch_dim=0,
                sequence_dim=1,
            ),
            options=compiler_kwargs,
        )

        # Write to a file for later use
        with open(self.zipfile_fname, "wb") as f:
            f.write(bitstream)

    def _extract_zipfile(self):
        """simply unpacks self.zipfile_fname into self.data_dir"""
        with zipfile.ZipFile(self.zipfile_fname, "r") as zip_ref:
            zip_ref.extractall(self.data_dir)

    def _compile_with_fm_to_fasmir(self, mapper_conf_kwargs={}):
        """Standard compilation sequence for FQIR, should eventually be grouped in FM"""
        try:
            from femtomapper.passman import (
                MapperState,
                MapperConf,
                PassMan,
                get_utm_inputs,
            )
        except ImportError:
            ImportError(
                "couldn't import femtomapper. This is a Femtosense-internal developer mode"
            )

        state = MapperState(fqir=self.fqir)
        conf = MapperConf(**mapper_conf_kwargs)
        passman = PassMan(conf)
        state = passman.do(state)

        self.fasmir = state.fasmir
        self.fmir = state.fmir

    @property
    def yamlfname(self):
        return os.path.join(self.data_dir, "metadata.yaml")

    @property
    def image_dir(self):
        return self.data_dir + "/" + MEM_IMAGE_FNAME_PRE

    def _extract_fasmir(self):
        """emit memory images from fasmir
        also optionally inserts debug stores first
        """

        try:
            from femtobehav.sim.runner import ProgState
        except ImportError:
            raise ImportError(
                "couldn't import femtobehav, needed to extract FASMIR. This is a Femtosense-internal developer mode"
            )

        # debug stores, calls the debugger
        if self.insert_debug_stores:
            try:
                from femtodriverpub.debugger import SPUDebugger
            except ImportError:
                raise ImportError(
                    "couldn't import debugger. This is a Femtosense-internal developer feature"
                )
            SPUDebugger.insert_debug_stores(self.fasmir)

        # dump the metadata yaml
        self.fasmir.get_yaml_metadata(self.yamlfname)

        # emit memory images
        basename = self.image_dir

        for cidx in range(len(self.fasmir.used_cores())):
            prog_state = ProgState.FromFASMIR(
                self.fasmir
            )  # just used to construct memory files, {cidx : femtobehav.fasmir.ProgState}
            self.mem_image_fnames = prog_state[cidx].save_packed_mems(
                basename, "_initial", use_core_str=True, encrypt=self.encrypt
            )

    def _add_fqir_meta(self):
        """adds FQIR metadata to yaml"""

        with open(self.yamlfname, "r") as f:
            meta = yaml.safe_load(f)

        input_padding = {}
        output_padding = {}

        for tproto in self.fqir.subgraphs["ARITH"].inputs:
            num_fasmir_words = meta["inputs"][tproto.name]["len_64b_words"]
            num_fasmir_els = packing.words_to_els(
                num_fasmir_words, precision=meta["inputs"][tproto.name]["precision"]
            )
            input_padding[tproto.name] = {
                "fqir": tproto.shape[0],
                "fasmir": num_fasmir_els,
            }

        for tproto in self.fqir.subgraphs["ARITH"].outputs:
            num_fasmir_words = meta["outputs"][tproto.name]["len_64b_words"]
            num_fasmir_els = packing.words_to_els(
                num_fasmir_words, precision=meta["outputs"][tproto.name]["precision"]
            )
            output_padding[tproto.name] = {
                "fqir": tproto.shape[0],
                "fasmir": num_fasmir_els,
            }

        meta["fqir_input_padding"] = input_padding
        meta["fqir_output_padding"] = output_padding

        with open(self.yamlfname, "w") as file:
            yaml.dump(meta, file, sort_keys=False)
