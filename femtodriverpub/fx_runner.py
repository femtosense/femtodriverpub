"""FemtoRunner for Femtocrux simulator"""
import logging
import copy
from femtorun import FemtoRunner
from femtocrux import CompilerClient, FQIRModel

logger = logging.getLogger(__name__)


class FXRunner(FemtoRunner):
    def __init__(
        self,
        fqir,
        batch_dim=0,
        sequence_dim=1,
        input_padding=None,
        output_padding=None,
    ):
        self.fqir = fqir
        self.batch_dim = batch_dim
        self.sequence_dim = sequence_dim

        # connect to docker
        self.client = CompilerClient()

        self.sim_report = None  # will fill in with run()

        super().__init__(input_padding, output_padding)

    def reset(self):
        # XXX HACK!
        # for some reason, we need the FS Quantizer to
        # de-floatify the inputs to the model
        # to make quantization have no effect, we hack the FQIR
        arith = self.fqir.subgraphs["ARITH"]
        for x in arith.inputs:
            x.quanta = 0  # scale = 2**quanta = 1 (I don't think this is the correct defn of quanta...)

        # get clean simulator object
        self.simulator = self.client.simulate(
            FQIRModel(
                self.fqir,
                batch_dim=self.batch_dim,
                sequence_dim=self.sequence_dim,
            )
        )

    def finish(self):
        pass

    def step(self, inputs):
        # we will just override run(), the docker's interface is more like that
        raise NotImplementedError(
            "FXRunner can only do run(), no fine-grained step() calls"
        )

    def run(self, input_val_timeseries):
        if len(input_val_timeseries) > 1:
            raise RuntimeError(
                "femtodriverpub's FXRunner can only handle models with single inputs"
            )

        fqir_outputs = self.fqir.subgraphs["ARITH"].outputs
        if len(fqir_outputs) > 1:
            raise RuntimeError(
                "femtodriverpub's FXRunner can only handle models with single outputs"
            )

        first_input_seq = next(iter(input_val_timeseries.values()))

        # tack on batch dim
        assert self.batch_dim == 0
        first_input_seq = first_input_seq.reshape((1, *first_input_seq.shape))

        # run the simulator
        # FX simulate just wants a numpy array, like the pytorch syntax
        client_outputs, self.sim_report = self.simulator.simulate(
            first_input_seq,
            # see FQIR quanta hack in FQIRModel initialization
            # a float is getting created somewhere, this needs a FS or FX patch
            # for now, we just use the quantization, but set the quanta so that
            # the effect of the Quantizer will just be to cast float -> int
            quantize_inputs=True,
            dequantize_outputs=False,
        )

        # have to figure out what the outputs are called
        first_name = next(iter([t.name for t in fqir_outputs]))

        assert self.batch_dim == 0
        client_outputs = client_outputs[0][0, :, :]  # remove batch dim

        output_always_valid_mask = {first_name: [True] * client_outputs.shape[0]}
        internal_vals = {}
        output_vals = {first_name: client_outputs}
        return output_vals, internal_vals, output_always_valid_mask
