from redis_plugin import CODE_TO_MSG

import redis
import argparse
import numpy as np
import time

class FakeRedisClient:
    def __init__(self, fake_hw_recv_vals):
        self.fake_hw_recv_vals = fake_hw_recv_vals
        self.r = redis.Redis()

    def _pop_until_empty(self, queue_name):
        read_data = []
        while True:
            val_bytes_or_none = self.r.rpop(queue_name)
            if val_bytes_or_none is None:
                break
            else:
                read_data.append(int(val_bytes_or_none))
        return read_data

    def run(self):

        shutdown = 0
        self.r.set('client_running', 'fake_redis_client')

        # loop forever, faking replies to read reqs, dumping writes in a black hole
        while not shutdown:

            # slow ourselves down a little
            time.sleep(.1)
            shutdown = int(self.r.get('shutdown_client'))
            if shutdown:
                print('received shutdown signal, exiting')

            # trash any writes
            write_req = self._pop_until_empty('write_req')
            if len(write_req) > 0:
                print('trashed some writes!')

            # reply to reads
            read_req = self._pop_until_empty('read_req')

            if len(read_req) > 0:
                print('found a read request')
                while len(read_req) != 4: # caught mid-transaction, try again
                    read_req += self._pop_until_empty('read_req')

                # unpack
                target_type_code, base_addr, final_addr, length = read_req
                msgtype = CODE_TO_MSG[target_type_code]
            
                # generate fake data
                if self.fake_hw_recv_vals is None:
                    num_words = length
                    if msgtype == 'axis':
                        num_words += 2

                    fake_data = np.ones((num_words,), dtype=np.uint32)

                    fake_data[:] = 0xfedcba9876543210
                    if msgtype == 'axis':
                        # fake 0 mailbox id, 'out' route is maxval=1f (it's ignored, however)
                        fake_data[0] = 0x1f
                        fake_data[1] = 0
                else: 
                    fake_data = self.fake_hw_recv_vals

                # send the fake reply back
                for val in fake_data:
                    assert len(fake_data) == length
                    self.r.lpush('read_reply', int(val))

    def __del__(self):
        self.r.set('client_running', 'none')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="set up a fake redis client")
    parser.add_argument("--valfile", default=None,
        help="fname with fake values, saved as numpy text")
    args = parser.parse_args()
    if args.valfile is not None:
        fake_vals = np.loadtxt(args.valfile)
    else:
        fake_vals = None
    client = FakeRedisClient(fake_vals)
    client.run()
    # kill by setting shutdown_client to 1

