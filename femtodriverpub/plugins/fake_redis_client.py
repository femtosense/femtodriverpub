from redis_plugin import CODE_TO_MSG

import redis
import argparse
import numpy as np
import time

import logging
import sys

logger = logging.getLogger(__file__)


class RedisRequest:
    READ = 0
    WRITE = 1
    CYCLE = 2

    APB = "apb"
    AXIS = "axis"
    SPU_TOP = "spu_top"
    HOST = "host"

    def __init__(self, req_type, msg_type, start_addr, end_addr, length):
        self.req_type = req_type
        self.msg_type = CODE_TO_MSG[msg_type]
        self.start_addr = start_addr
        self.end_addr = end_addr
        self.length = length
        self.data = None

    def set_data(self, data):
        assert len(data) == self.length
        assert self.req_type == self.WRITE


class FakeRedisClient:
    def __init__(self, fake_hw_recv_vals):
        self.fake_hw_recv_vals = fake_hw_recv_vals
        self.r = redis.Redis()
        self.data_q = []

    def _pop_until_empty(self, queue_name):
        read_data = []
        while True:
            val_bytes_or_none = self.r.lpop(queue_name)
            if val_bytes_or_none is None:
                break
            else:
                read_data.append(int(val_bytes_or_none))
        return read_data

    def _send_data(self, data, queue_name):
        for d in data:
            self.r.rpush(queue_name, int(d))
            logger.debug(f"sending {d}")

    def get_data_from_redis(self, num_elems):
        """Get num_elems elements from redis"""
        while len(self.data_q) < num_elems:
            self.data_q += self._pop_until_empty("req")

        data = self.data_q[:num_elems]
        self.data_q = self.data_q[num_elems:]
        return data

    def send_data_to_redis(self, data):
        logger.debug(f"sending reply {data}")
        self._send_data(data, "reply")

    def parse_msg(self):
        """blocking waits until a full message is received"""
        req = self.get_data_from_redis(5)
        req = RedisRequest(*req)
        if req.req_type == RedisRequest.WRITE:
            data = self.get_data_from_redis(req.length)
            req.set_data(data)
        return req

    def wait_for_redis(self):
        """waits for redis server to come online"""
        while True:
            try:
                self.r.ping()
                break
            except redis.ConnectionError:
                time.sleep(1)
                logger.info("waiting for redis-server to come online")

    def run(self):
        shutdown = 0

        self.wait_for_redis()
        self.r.set("client_running", "fake_redis_client")

        # loop forever, faking replies to read reqs, dumping writes in a black hole
        while not shutdown:
            # slow ourselves down a little
            time.sleep(0.1)
            shutdown = self.r.get("client_shutdown")
            if shutdown is None:
                logger.debug(
                    "no client_shutdown value, waiting for redis_plugin to start sending commands"
                )
                continue

            shutdown = int(shutdown)
            if shutdown:
                logger.info("received shutdown signal, exiting")

            logger.debug(f"data_q {self.data_q}")
            req = self.parse_msg()
            logger.debug(
                f"got msg type={req.req_type} {req.msg_type} addr={req.start_addr},{req.end_addr} len={req.length}, data={req.data}"
            )

            if req.req_type == RedisRequest.READ:
                logger.info("Got read request, replying with fake data")
                reply = []
                if req.msg_type == RedisRequest.AXIS:
                    # fake 0 mailbox id, 'out' route is maxval=1f (it's ignored, however)
                    reply.append(0x1F)
                    reply.append(0)

                fake_data = np.zeros(req.length, dtype=np.uint32)
                if self.fake_hw_recv_vals is None:
                    fake_data[:] = 0x76543210
                else:
                    # make sure we reply with as many words as were requested
                    truncated_data = self.fake_hw_recv_vals[: req.length]
                    fake_data[: len(truncated_data)] = truncated_data

                reply.extend(fake_data)
                # send reply
                self.send_data_to_redis(reply)

    def __del__(self):
        try:
            self.r.set("client_running", "none")
        except redis.ConnectionError:
            logger.warning(
                "fake redis client lost connection to redis server before __del__"
            )


if __name__ == "__main__":
    # override logger so that it prints to stdout
    # when this is run as a subprocess
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    logging.basicConfig(handlers=[stdout_handler])
    logger = logging.getLogger(__file__)

    parser = argparse.ArgumentParser(description="set up a fake redis client")
    parser.add_argument(
        "--valfile", default=None, help="fname with fake values, saved as numpy text"
    )
    args = parser.parse_args()
    if args.valfile is not None:
        fake_vals = np.loadtxt(args.valfile)
    else:
        fake_vals = None
    client = FakeRedisClient(fake_vals)
    client.run()
    # kill by setting shutdown_client to 1
