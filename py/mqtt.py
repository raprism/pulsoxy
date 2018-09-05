# coding: utf-8
import sys
import logging
import asyncio

import pandas as pd

from hbmqtt.client import MQTTClient, ClientException
from hbmqtt.mqtt.constants import QOS_0

# local
import oxy


class SkipTimeouts(logging.Filter):
    def filter(self, rec):
        if (rec.levelno == logging.INFO and rec.msg.startswith('poll') and
                rec.msg.endswith(': timeout') and
                rec.args[1] - rec.args[0] < 10):
            ## TODO here rec.args were not set as awaited
            #print(rec.args)
            return False  # hide this record
        return True


if True:
    #if False:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
else:
    logging.basicConfig(stream=open('log.out', 'w'), level=logging.DEBUG)
logger = logging.getLogger(__file__)
#logger.addFilter(SkipTimeouts())

server = 'mqtt://127.0.0.1/'
data_fn = 'data/demo2.csv'
topic = 'outTopic'


async def pub_produce(queue, n=None):
    df = oxy.Oxy(data_fn).data[oxy.Oxy.cols]  ## no generated columns
    dlen = 100
    assert len(df) > dlen
    i = j = 0
    df['time_ms'] -= df.time_ms[0]
    dt = int(df.time_ms.diff().mean())
    t_orig = df.time_ms.copy()
    while True:
        if j >= len(df) - dlen:
            j = len(df) % dlen
            df['time_ms'] += t_orig.iloc[-j] + dt
        msg = df.iloc[j:j+dlen]\
                .astype(str)\
                .apply(lambda r: ';'.join(r), axis=1)\
                .to_frame()\
                .apply(lambda c: '\n'.join(c))[0]
        await queue.put(msg.encode('ascii'))
        j += dlen
        i += 1
        if n is not None and i >= n:
            await queue.join()
            logger.info('Will exit in 1 second ...')
            await asyncio.sleep(1)
            break
        await asyncio.sleep(dt / 1000 * dlen)


async def pub_consume(queue):
    cli = MQTTClient()
    await cli.connect(server)
    try:
        while True:
            # wait for an item from the producer
            msg = await queue.get()
            # publish
            await cli.publish(topic, msg, qos=QOS_0)
            logger.info("messages published")
            # Notify the queue that the item has been processed
            queue.task_done()
    except asyncio.CancelledError:
        logger.debug('CancelledError')
        ## TODO: still error 'Task was destroyed but it is pending!'
        await cli.disconnect()
        queue.task_done()
        logger.info('Cancelled gracefully.')
    except ClientException as ce:
        logger.error("Client exception: %s" % ce)


async def sub_produce(queue, n=None):
    cli = MQTTClient()
    await cli.connect(server)
    await cli.subscribe([(topic, QOS_0)])
    try:
        i = 0
        while True:
            msg = await cli.deliver_message()
            packet = msg.publish_packet
            data = packet.payload.data
            logger.info("== %s ==" % (packet.variable_header.topic_name))
            logger.debug(data.decode('ascii'))
            ## TODO break on special message
            await queue.put(data)
            i += 1
            if n is not None and i >= n:
                break
        await cli.disconnect()
    except ClientException as ce:
        logger.error("Client exception: %s" % ce)

def _test_action(buffer, keep=None):
    if keep is not None:

        if isinstance(keep, pd.DataFrame):
            logger.info(keep.shape)
        elif isinstance(keep, oxy.Oxy):
            logger.info(keep.data.shape)
            # update evaluation
            keep.eval()
            #logger.info(pd.concat([keep.data.head(), keep.data.tail()]))
    if False:
        return buffer
    else:
        oo = oxy.Oxy(buffer, do_eval=False)
        return oo


async def sub_consume(queue, action,
                      n=None,
                      keep=None,
                      keep_lim=400):
    def _concat(df, ret):
        #cols = oxy.Oxy.cols # raw data only
        #ret = pd.concat([keep[cols], ret]).reset_index(drop=True)
        return pd.concat([df, ret.data]).reset_index(drop=True)
    i = 0
    while True:
        # wait for an item from the producer
        buffer = await queue.get()
        # process the item
        ret = action(buffer, keep=keep)
        assert isinstance(ret, oxy.Oxy)
        if keep is not None:
            if isinstance(keep, pd.DataFrame):
                ret.data = _concat(keep, ret)
                keep = ret.data.iloc[-keep_lim:]
            elif isinstance(keep, oxy.Oxy):
                ret.data = _concat(keep.data, ret)
                keep.update(ret.data.iloc[-keep_lim:])
        logger.info(pd.concat([ret.data.head(), ret.data.tail()]))
        # Notify the queue that the item has been processed
        queue.task_done()
        i += 1
        if n is not None and i >= n:
            break


async def run_publisher(n=None):
    queue = asyncio.Queue()
    # schedule the consumer
    consumer = asyncio.ensure_future(pub_consume(queue))
    # run the producer and wait for completion
    await pub_produce(queue, n)
    # wait until the consumer has processed all items
    await queue.join()
    # the consumer is still awaiting for an item, cancel it
    consumer.cancel()


async def run_subscription(consumer_action,
                           produce_n=None,
                           consume_n=None,
                           consume_keep=None):
    queue = asyncio.Queue()
    # schedule the consumer
    consumer = asyncio.ensure_future(sub_consume(
        queue, consumer_action, n=consume_n, keep=consume_keep))
    # run the producer and wait for completion
    await sub_produce(queue, n=produce_n)
    # wait until the consumer has processed all items
    await queue.join()
    # the consumer is still awaiting for an item, cancel it
    consumer.cancel()


if __name__ == '__main__':

    ## loop count
    #n = n_p = 10
    ## ... infinite
    n = None
    n_p = None
    #
    #df_init = pd.DataFrame(columns=oxy.Oxy.cols)
    oo_init = oxy.Oxy()

    loop = asyncio.get_event_loop()

    try:
        if '-p' in sys.argv[1:]:
            loop.run_until_complete(run_publisher(n_p))
        else:
            future = run_subscription(
                consumer_action=_test_action,
                consume_n=n,
                produce_n=n_p,
                consume_keep=oo_init)
            loop.run_until_complete(future)
    finally:
        loop.close()
